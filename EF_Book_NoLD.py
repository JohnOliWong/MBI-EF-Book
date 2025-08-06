'''
We removed the optimal transport loss L_{OT}
'''

import torch
import torch.nn.functional as F
from torch import nn
import random

from Encoder_EF import Encoder_EF
from Encoder_Common import Encoder_Common


class EGA(nn.Module):
	def __init__(self, emb_size, num_heads=4, drop_out=0.3):
		super().__init__()

		self.attn = nn.MultiheadAttention(
			embed_dim=emb_size,
			num_heads=num_heads,
			dropout=drop_out,
			batch_first=True,
		)

	def forward(self, x, y):
		# x is the dominant modality, e.g. EEG
		attn_output, _ = self.attn(
			query=x,
			key=y,
			value=y,
		)

		return attn_output


class Quantization(nn.Module):
	'''
	Args:

	cont_features: input continuous concatenated EEG-fNIRS features [B, T, E]
	quan_features: quantized vectors [B, T, E]
	codebook: a dictionary of prototype vectors [D, E]
	N: number of times each codeword is matched [D,]
	m: summation of continuous features matched to the current codeword [D, E]
	status: reflect if a codeword is active or not [D,]
	'''
	def __init__(self, dict_len, emb_size, decay=0.99, quan_weight=1.0, ot_weight=1.0, threshold=60):
		super().__init__()
		self.dict_len = dict_len
		self.emb_size = emb_size
		self.decay = decay
		self.quan_weight = quan_weight
		self.ot_weight = ot_weight
		self.threshold = threshold

		# initiate subject-specific, modality-invariant codebook
		self.codebook = nn.Parameter(torch.randn(dict_len, emb_size))
		self.register_buffer('N', torch.ones(dict_len))
		self.register_buffer('m', self.codebook.clone())
		self.register_buffer('status', torch.zeros(dict_len))

	def forward(self, cont_features):
		# compute L2-distance between each feature-codeword pair
		distances = torch.cdist(cont_features, self.codebook)
		quan_idx = torch.argmin(distances, dim=-1) # [2 * B]
		quan_token = self.codebook.data[quan_idx] # [2 * B, E]
		quan_token = cont_features + (quan_token - cont_features).detach()
		
		# codebook variables are only updated during training
		usage = None
		if self.training:
			usage = self.codebook_update(cont_features, quan_idx)

		vq_loss = F.mse_loss(cont_features.detach(), quan_token)
		commit_loss = F.mse_loss(cont_features, quan_token.detach())
		quan_loss = vq_loss + commit_loss
		ot_loss = self.ot_loss(cont_features, quan_idx)
		codebook_loss = self.quan_weight * quan_loss + self.ot_weight * ot_loss

		if usage is not None:
			return {'usage': usage, 'quan_token': quan_token, 'quan_loss': quan_loss, 'ot_loss': ot_loss}
		else:
			return {'quan_token': quan_token, 'quan_loss': quan_loss, 'ot_loss': ot_loss}
	
	def codebook_update(self, features, indices):
		with torch.no_grad():
			features = features.float()
			# converts discrete codeword indices to sparse matrix
			one_hot = F.one_hot(indices, num_classes=len(self.codebook)).float() # [2 * B, D]
			counts = one_hot.sum(dim=[0]) # [D]
			matched = (counts > 0)
			usage = torch.count_nonzero(matched)
			
			if matched.any():
				# summation of fine-grained representations matched to the corresponding codeword
				sum_features = torch.einsum('nc,nd->dc', features, one_hot) # [D, E]
				self.N[matched] = self.decay * self.N[matched] + (1-self.decay) * counts[matched]
				self.m[matched] = self.decay * self.m[matched] + (1-self.decay) * sum_features[matched, :]
				self.codebook.data[matched] = self.m[matched] / self.N[matched].unsqueeze(1)
				self.status[matched] = 0
			
			inactive = ~matched
			self.status[inactive] += 1
			if (self.status >= self.threshold).any():
				self.codeword_replace()
			return usage

	def codeword_replace(self):
		inactive = (self.status >= self.threshold)
		if inactive.any() and (self.N > 1).any():
			active_idx = torch.where(self.N > 1)[0]
			random_idx = random.choice(active_idx.cpu().numpy())

			replace_codeword = self.codebook.data[random_idx].clone()
			self.codebook.data[inactive] = replace_codeword
			self.N[inactive] = 1
			self.m[inactive] = self.codebook.data[inactive].clone()
			self.status[inactive] = 0

	def ot_loss(self, features, indices):
		one_hot = F.one_hot(indices, num_classes=len(self.codebook)).float()
		# compute the probability of each codeword being matched
		codebook_probs = one_hot.mean(dim=[0])
		# the ideal situation is all codewords have the same chance to be used
		target_probs = torch.ones_like(codebook_probs) / len(self.codebook)
		# utilize KL-divergence to make the actual codeword distribution align with the ideal distribution
		ot_loss = F.kl_div(
			input=torch.log(codebook_probs + 1e-6),
			target=target_probs,
			reduction='batchmean'
		)
		return ot_loss


class AttentionFusion(nn.Module):
	def __init__(self, emb_size):
		super(AttentionFusion, self).__init__()
		self.weight = nn.Parameter(torch.randn(emb_size, 1), requires_grad=True)
		self.softmax = nn.Softmax(dim=-1)
		self.alpha = None

	def forward(self, out):
		o = torch.cat([i @ self.weight for i in out], dim=-1) # [16, 2] or [16,]
		if o.dim() == 1:
			o = o.unsqueeze(1)
		self.alpha = self.softmax(o)
		outputs = sum([i * self.alpha[:, index].unsqueeze(1) for index, i in enumerate(out)])
		return outputs


class Classifier(nn.Module):
	def __init__(self, emb_size, num_class):
		super().__init__()
		
		self.weight_eeg = AttentionFusion(emb_size)
		self.weight_nirs = AttentionFusion(emb_size)
		self.weight_fusion = AttentionFusion(emb_size)
		self.weights = nn.Parameter(torch.Tensor([1., 1., 0.5]), requires_grad=True)

		self.projector = nn.Sequential(
			nn.Linear(emb_size, emb_size // 2),
			nn.ReLU(),
			nn.Linear(emb_size // 2, num_class),
			nn.ReLU(),
		)
	
	def forward(self, eeg_token, nirs_token, quan_eeg, quan_nirs):
		eeg_fusion = self.weight_eeg([eeg_token, quan_eeg])
		nirs_fusion = self.weight_nirs([nirs_token, quan_nirs])
		sum_fusion = eeg_token + nirs_token + quan_eeg + quan_nirs

		w1 = torch.exp(self.weights[0]) / torch.sum(torch.exp(self.weights))
		w2 = torch.exp(self.weights[1]) / torch.sum(torch.exp(self.weights))
		w3 = torch.exp(self.weights[2]) / torch.sum(torch.exp(self.weights))

		outputs = w1 * self.projector(eeg_fusion) + w2 * self.projector(nirs_fusion) + w3 * self.projector(sum_fusion)
		
		return outputs


class Cosine_Loss(nn.Module):
	def __init__(self):
		super(Cosine_Loss, self).__init__()

	def forward(self, x, y, threshold=0.2, eps=1e-6):
		dot = torch.sum(x * y, dim=1)
		norm_x = torch.norm(x, p=2, dim=1)
		norm_y = torch.norm(y, p=2, dim=1)
		cosine_sim = dot / (norm_x * norm_y + eps)
		constraint = torch.relu(torch.abs(cosine_sim) - threshold)
		return torch.mean(constraint ** 2)


class EF_Book(nn.Module):
	def __init__(self, dict_len, emb_size, num_class, mode, threshold, device):
		super().__init__()
		self.emb_size = emb_size
		if mode == 0 or mode == 1:
			self.EEG_Width = 4000
			self.NIRS_Width = 200
		elif mode == 2:
			self.EEG_Width = 2000
			self.NIRS_Width = 100
		self.num_TConv = 4
		self.num_SConv = 8
		
		self.ef_conv = Encoder_EF(depth=4, query_size=emb_size, key_size=emb_size, value_size=emb_size, emb_size=emb_size, num_heads=4, expansion=2, conv_dropout=0.3,
                 				  self_dropout=0.3, cross_dropout=0.3, cls_dropout=0.5, num_classes=num_class, mode=mode, device=device)
		self.eeg_common_conv = Encoder_Common(num_class, emb_size, T_Width=self.EEG_Width, S_Height=30, num_TConv=self.num_TConv, num_SConv=self.num_SConv)
		self.nirs_common_conv = Encoder_Common(num_class, emb_size, T_Width=self.NIRS_Width, S_Height=72, num_TConv=self.num_TConv, num_SConv=self.num_SConv)

		self.eeg_quantizer = Quantization(dict_len, self.emb_size, threshold=threshold)
		self.nirs_quantizer = Quantization(dict_len, self.emb_size, threshold=threshold)
		self.fusion_quantizer = Quantization(dict_len, self.emb_size, threshold=threshold)
		self.cosine_loss = Cosine_Loss()
		self.ega = EGA(emb_size)
		self.classifier = Classifier(emb_size, num_class)

	def forward(self, eeg, nirs, last_batch):
		eeg_p_token, nirs_p_token = self.ef_conv(eeg, nirs) # [16, 128] = [B, E]
		eeg_s_token = self.eeg_common_conv(eeg) # [16, 128] = [B, E]
		nirs_s_token = self.nirs_common_conv(nirs)
		disentangling_loss = self.cosine_loss(eeg_s_token, eeg_p_token) + self.cosine_loss(nirs_s_token, nirs_p_token)
		
		# EEG private codebook
		eeg_quan_outputs = self.eeg_quantizer(eeg_p_token)
		eeg_quan_usage = None
		if len(eeg_quan_outputs) == 4:
			eeg_quan_usage = eeg_quan_outputs['usage']
		quan_eeg, eeg_quan_loss = eeg_quan_outputs['quan_token'], eeg_quan_outputs['quan_loss'] + eeg_quan_outputs['ot_loss']

		# fNIRS private codebook
		nirs_quan_outputs = self.nirs_quantizer(nirs_p_token)
		nirs_quan_usage = None
		if len(nirs_quan_outputs) == 4:
			nirs_quan_usage = nirs_quan_outputs['usage']
		quan_nirs, nirs_quan_loss = nirs_quan_outputs['quan_token'], nirs_quan_outputs['quan_loss'] + nirs_quan_outputs['ot_loss']
		
		# EEG-fNIRS shared codebook
		batch_size = eeg_p_token.shape[0]
		nirs_s_token = self.ega(eeg_s_token, nirs_s_token)
		fusion_features = torch.cat([eeg_s_token, nirs_s_token], dim=0)
		fusion_quan_outputs = self.fusion_quantizer(fusion_features)
		fusion_quan_usage = None
		if len(fusion_quan_outputs) == 4:
			fusion_quan_usage = fusion_quan_outputs['usage']
		quan_fusion, fusion_quan_loss = fusion_quan_outputs['quan_token'], fusion_quan_outputs['quan_loss'] + fusion_quan_outputs['ot_loss']
		quan_fusion_eeg, quan_fusion_nirs = quan_fusion[:batch_size, :], quan_fusion[batch_size:, :]
		if last_batch and eeg_quan_usage is not None:
			print(f'Codebook Usage: EEG {eeg_quan_usage} fNIRS {nirs_quan_usage} Shared {fusion_quan_usage}')
		
		# classification
		eeg_token = eeg_p_token
		nirs_token = nirs_p_token
		quan_eeg = quan_eeg + quan_fusion_eeg
		quan_nirs = quan_nirs + quan_fusion_nirs
		outputs = self.classifier(eeg_token, nirs_token, quan_eeg, quan_nirs)
		codebook_loss = eeg_quan_loss + nirs_quan_loss + 0.5 * fusion_quan_loss

		return {
			'outputs': outputs,
			'loss': 0.15 * codebook_loss,
		}
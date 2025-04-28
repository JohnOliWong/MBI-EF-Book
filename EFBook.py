import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange
import random

from DualConvLayer import TemporalConvLayer

# positional embedding
class PositionalEmbedding(nn.Module):
	def __init__(self, channels, emb_size, device):
		super().__init__()
		self.channels = channels + 1
		self.pos_emb = nn.Parameter(torch.randn(size=(1, self.channels, emb_size), dtype=torch.float32, device=device),
									requires_grad=True)

	def forward(self, x):
		x = self.pos_emb + x
		return x


# induce modality labels, eeg = 1, fnirs = 0, this is to help the model to distinguish data type
class ModalityTypeEmbedding(nn.Module):
	def __init__(self, emb_size, token_type_idx=1):
		super().__init__()
		self.token_type_idx = token_type_idx
		self.type_embedding_layer = nn.Embedding(2, emb_size)

	def forward(self, x, mask):
		# x.shape = b, 1 + eeg_tokens + 1 + nirs_tokens, emb_size
		# mask.shape = [1 + eeg_tokens, 1 + nirs_tokens]
		b, _, emb_size = x.shape
		modality_type_emb = torch.ones(b, mask[0] + mask[1], dtype=torch.long, device=x.device) # torch.ones generate a matrix consists of "1"s
		modality_type_emb[:, mask[0]::] = 0 # mask[0]:: = from mask[0] to the end
		type_emb = self.type_embedding_layer(modality_type_emb)
		x = x + type_emb
		return x


# insert learnable cls tokens
class AddClsToken(nn.Module):
	def __init__(self, emb_size):
		super().__init__()
		self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size), requires_grad=True)

	def forward(self, x):
		self.cls_token = self.cls_token.to(x.device)
		x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1) # cls token is repeated batch_size times and concatenated to the start of x
		return x


# inter-series interaction
class MultiHeadAttention(nn.Module):
	def __init__(self, query_size, key_size, value_size, emb_size, num_heads, dropout, bias=False):
		super().__init__()
		self.emb_size = emb_size
		self.proj_dim = self.emb_size
		self.num_heads = num_heads
		self.queries = nn.Linear(query_size, self.proj_dim, bias=bias)
		self.keys = nn.Linear(key_size, self.proj_dim, bias=bias)
		self.values = nn.Linear(value_size, self.proj_dim, bias=bias)
		self.att_drop = nn.Dropout(dropout)
		self.projection = nn.Sequential(
			nn.Linear(self.proj_dim, emb_size, bias=bias),
			nn.Dropout(dropout)
		)
		self.attention_weights = None

	def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
		queries = rearrange(self.queries(query), "b n (h d) -> b h n d", h=self.num_heads)
		keys = rearrange(self.keys(key), "b n (h d) -> b h n d", h=self.num_heads)
		values = rearrange(self.values(value), "b n (h d) -> b h n d", h=self.num_heads)
		energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
		if mask is not None:
			fill_value = torch.finfo(torch.float32).min
			energy.mask_fill(~mask, fill_value)

		scaling = self.emb_size ** (1 / 2)
		self.attention_weights = F.softmax(energy / scaling, dim=-1)
		att = self.att_drop(self.attention_weights)
		out = torch.einsum('bhal, bhlv -> bhav ', att, values)
		out = rearrange(out, "b h n d -> b n (h d)")
		out = self.projection(out)
		return out


# intra-series modeling
# when called, query = key = value
class MultiHeadSelfAttention(nn.Module):
	def __init__(self, query_size, key_size, value_size, emb_size, num_heads, dropout, bias=False):
		super().__init__()
		self.proj_dim = emb_size
		self.num_heads = num_heads
		self.queries = nn.Linear(query_size, self.proj_dim, bias=bias)
		self.keys = nn.Linear(key_size, self.proj_dim, bias=bias)
		self.values = nn.Linear(value_size, self.proj_dim, bias=bias)
		self.att_drop = nn.Dropout(dropout)
		self.projection = nn.Sequential(
			nn.Linear(self.proj_dim, emb_size, bias=bias),
			nn.Dropout(dropout)
		)
		self.attention_weights = None
		self.scaling = emb_size ** (1 / 2)

	def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
		queries = rearrange(self.queries(query), "b n (h d) -> b h n d", h=self.num_heads)
		keys = rearrange(self.keys(key), "b n (h d) -> b h n d", h=self.num_heads)
		values = rearrange(self.values(value), "b n (h d) -> b h n d", h=self.num_heads)
		energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
		if mask is not None:
			fill_value = torch.finfo(torch.float32).min
			energy.mask_fill(~mask, fill_value)

		self.attention_weights = F.softmax(energy / self.scaling, dim=-1)
		att = self.att_drop(self.attention_weights)
		out = torch.einsum('bhal, bhlv -> bhav ', att, values)
		out = rearrange(out, "b h n d -> b n (h d)")
		out = self.projection(out)
		return out


# FFN
# introduces non-linearity and feature augmentation
class FeedForwardBlock(nn.Module):
	def __init__(self, emb_size, expansion=2, dropout=0.5):
		super().__init__()
		self.feed_forward = nn.Sequential(
			nn.Linear(emb_size, expansion * emb_size),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(expansion * emb_size, emb_size),
			nn.Dropout(dropout)
		)

	def forward(self, x):
		x = self.feed_forward(x)
		return x


# self-attention + residual connection
class SelfEncoderBlock(nn.Module):
	def __init__(self, query_size, key_size, value_size, emb_size, num_heads=4, forward_expansion=2, dropout=0.5):
		super(SelfEncoderBlock, self).__init__()

		self.attention = MultiHeadSelfAttention(query_size, key_size, value_size, emb_size, num_heads, dropout)
		self.feed_forward = FeedForwardBlock(emb_size, expansion=forward_expansion, dropout=dropout)

		self.norm1 = nn.LayerNorm(emb_size)
		self.norm2 = nn.LayerNorm(emb_size)

	def forward(self, x):
		# x is dominant modality
		res = x
		x = self.norm1(x)
		y = self.attention(x, x, x) # q = k = v
		y = y + res

		res = y
		y2 = self.norm2(y)
		y2 = self.feed_forward(y2)
		y2 = y2 + res

		return y2


# cross-attention + residual connection
class CrossEncoderBlock(nn.Module):
	def __init__(self, query_size, key_size, value_size, emb_size, num_heads=4, forward_expansion=2, dropout=0.5):
		super(CrossEncoderBlock, self).__init__()

		self.attention = MultiHeadAttention(query_size, key_size, value_size, emb_size, num_heads, dropout)
		self.feed_forward = FeedForwardBlock(emb_size, expansion=forward_expansion, dropout=dropout)

		self.norm1 = nn.LayerNorm(emb_size)
		self.norm2 = nn.LayerNorm(emb_size)

	def forward(self, x, y):
		# x is dominant modality
		res = x
		x, y = self.norm1(x), self.norm2(y)
		y1 = self.attention(x, y, y) # q from one modality, k and v from the other
		y1 = y1 + res

		res = y1
		y2 = self.norm2(y1)
		y2 = self.feed_forward(y2)
		y2 = y2 + res

		return y2


# intra-modal attention
class TransformerSelfEncoder(nn.Module):
	def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
				 device):
		super(TransformerSelfEncoder, self).__init__()
		self.blks = nn.Sequential()
		self.attention_weights = [None] * depth
		self.add_token = AddClsToken(emb_size)
		self.positional_embedding = PositionalEmbedding(channels, emb_size, device)
		for i in range(depth):
			self.blks.add_module("block" + str(i),
								 SelfEncoderBlock(query_size, key_size, value_size, emb_size, num_heads, expansion,
												  dropout))

	def forward(self, x, mask=None):
		x = self.add_token(x)
		x = self.positional_embedding(x)
		for i, blk in enumerate(self.blks):
			x = blk(x)
			self.attention_weights[i] = blk.attention.attention_weights
		return x

	@property
	def self_attention_weights(self):
		return self.attention_weights


# cross-modal attention
# no need of inserting cls token and positional embedding, since this step has been done in self-encoder
# class TransformerCrossEncoder(nn.Module):
# 	def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
# 				 device):
# 		super(TransformerCrossEncoder, self).__init__()

# 		self.blks = nn.Sequential() # blks is like a multi-layer container for stacking encoders
# 		self.attention_weights = [None] * depth # store attention weights of each depth

# 		for i in range(depth):
# 			self.blks.add_module("block" + str(i),
# 								 CrossEncoderBlock(query_size, key_size, value_size, emb_size, num_heads, expansion,
# 												   dropout)) # cross-encoder

# 	def forward(self, x, y):
# 		for i, blk in enumerate(self.blks):
# 			x = blk(x, y)
# 			self.attention_weights[i] = blk.attention.attention_weights
# 		return x

# 	@property
# 	def cross_attention_weights(self):
# 		return self.attention_weights


# concatenate input from both modalities, then start attention mechanism in an intra-modal fashion
# class TransformerCatEncoder(nn.Module):
# 	def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
# 				 device):
# 		super(TransformerCatEncoder, self).__init__()
# 		self.modality_embedding = ModalityTypeEmbedding(emb_size)
# 		self.blks = nn.Sequential()
# 		self.attention_weights = [None] * depth

# 		for i in range(depth):
# 			self.blks.add_module("block" + str(i),
# 								 SelfEncoderBlock(query_size, key_size, value_size, emb_size, num_heads, expansion,
# 												  dropout)) # self-encoder

# 	def forward(self, x, y, mask=None):
# 		context = torch.cat([x, y], dim=1)
# 		context = self.modality_embedding(context, mask)
# 		for i, blk in enumerate(self.blks):
# 			context = blk(context)
# 			self.attention_weights[i] = blk.attention.attention_weights
# 		return context

# 	@property
# 	def self_attention_weights(self):
# 		return self.attention_weights


# intra- and inter- modality encoder
class Transformer(nn.Module):
	def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, device,
				 self_dropout, cross_dropout):
		super().__init__()
		self.eeg_nirs_temporal_spatial_attention_weights = None
		self.eeg_temporal_spatial_attention_weights = None
		self.eeg_temporal_attention_weights = None
		self.eeg_spatial_attention_weights = None
		self.nirs_spatial_attention_weights = None
		self.mask = [channels[0] + 1, channels[1] + 1]

		self.eeg_temporal_encoder = TransformerSelfEncoder(depth, query_size, key_size, value_size, emb_size,
														   num_heads, channels[0], expansion, cross_dropout, device)

		self.nirs_temporal_encoder = TransformerSelfEncoder(depth, query_size, key_size, value_size, emb_size,
															num_heads, channels[1], expansion, cross_dropout, device)

	def forward(self, temporal_eeg, temporal_nirs):
		eeg_temporal_outputs = self.eeg_temporal_encoder(temporal_eeg)
		# print(eeg_temporal_outputs.shape)
		nirs_temporal_outputs = self.nirs_temporal_encoder(temporal_nirs)
		# print(nirs_temporal_outputs.shape)

		return eeg_temporal_outputs[:, 0], nirs_temporal_outputs[:, 0]


	@property
	def get_eeg_attention_weights(self):
		return [self.eeg_temporal_attention_weights, self.eeg_spatial_attention_weights]

	@property
	def get_nirs_spatial_attention_weights(self):
		return [self.nirs_spatial_attention_weights, ]

	@property
	def get_cross_attention_weights(self):
		return [self.eeg_nirs_temporal_spatial_attention_weights, self.eeg_temporal_spatial_attention_weights]


class AttentionFusion(nn.Module):
	def __init__(self, emb_size):
		super(AttentionFusion, self).__init__()
		self.weight = nn.Parameter(torch.randn(emb_size, 1), requires_grad=True)
		self.softmax = nn.Softmax(-1)

	def forward(self, out):
		o = torch.cat([i @ self.weight for i in out], dim=-1)
		alpha = self.softmax(o)
		outputs = sum([i * alpha[:, index].unsqueeze(1) for index, i in enumerate(out)])
		return outputs


class Quantization(nn.Module):
	'''
	args:
	cont_features: [16, 252, 64]
	codebook: [512, 64]
	N: [512,]
	m: [1, 512, 64]
	status: [512,]
	'''
	def __init__(self, dict_len=512, emb_size=64, decay=0.99):
		super().__init__()
		self.codebook = nn.Parameter(torch.randn(dict_len, emb_size))
		self.decay = decay
		
		self.register_buffer('N', torch.ones(dict_len))
		self.register_buffer('m', self.codebook.clone())
		self.register_buffer('status', torch.zeros(dict_len))
		
	def forward(self, cont_features):
		distances = torch.cdist(cont_features, self.codebook)  # [16, 252, 512]
		quan_idx = torch.argmin(distances, dim=-1)  # [16, 252]
		quan_token = self.codebook.data[quan_idx]  # [16, 252, 64]
		quan_token = cont_features + (quan_token - cont_features).detach()
		
		# codebook varables are only updated during training
		if self.training:
			self.codebook_update(cont_features, quan_idx)
		
		codebook_loss = F.mse_loss(quan_token.detach(), cont_features)
		return quan_token, codebook_loss
	
	def codebook_update(self, features, indices):		
		with torch.no_grad():
			one_hot = F.one_hot(indices, num_classes=len(self.codebook)).float()  # [16, 252, 512]
			counts = one_hot.sum(dim=[0,1])  # [512]
			matched = (counts > 0)
			
			if matched.any(): # .any() returns bool variables
				sum_features = torch.einsum('bsd,bs->d', features, one_hot.sum(dim=1))
				self.N[matched] = self.decay * self.N[matched] + (1-self.decay) * counts[matched]
				self.m[matched] = self.decay * self.m[matched] + (1-self.decay) * sum_features[matched]
				self.codebook.data[matched] = self.m[matched] / self.N[matched].unsqueeze(1)
				self.status[matched] = 0
			
			inactive = ~matched
			self.status[inactive] += 1
			if (self.status >= 100).any():
				self.codeword_replace()

	def codeword_replace(self):
		inactive = (self.status >= 100)
		if inactive.any() and (self.N > 1).any():
			active_idx = torch.where(self.N > 1)[0]
			random_idx = random.choice(active_idx)
			# replace_idx = torch.randint(0, len(active_idx), (inactive.sum(),))
			
			self.codebook.data[inactive] = self.codebook.data[random_idx]
			self.N[inactive] = 1
			self.m[inactive] = self.codebook.data[inactive]
			self.status[inactive] = 0


class Classifier(nn.Module):
	def __init__(self, emb_size=64, num_classes=2):
		super().__init__()

		self.eeg_weight = nn.Parameter(torch.ones(1))
		self_nirs_weight = nn.Parameter(torch.ones(1))

		self.eeg_proj = nn.Linear(emb_size * 2, emb_size)
		self.nirs_proj = nn.Linear(emb_size * 2, emb_size)

		self.classifier = nn.Sequential(
			nn.Linear(emb_size * 2, 128),
			nn.ReLU(),
			nn.Linear(128, num_classes)
		)

	def forward(self, batch_eeg, batch_nirs, quan_eeg, quan_nirs):
		eeg_cat = torch.cat([batch_eeg, quan_eeg], dim=-1)
		eeg_pooled = eeg_cat.mean(dim=1)
		eeg_feat = self.eeg_proj(eeg_pooled)

		nirs_cat = torch.cat([batch_nirs, quan_nirs], dim=-1)
		nirs_pooled = nirs_cat.mean(dim=1)
		nirs_feat = self.nirs_proj(nirs_pooled)

		weights = torch.softmax(torch.stack([self.eeg_weight, self.nirs_weight]), dim=0)
		fused = weights[0] * eeg_feat + weights[1] * nirs_feat

		logits = self.classifier(torch.cat([eeg_feat, nirs_feat], dim=-1))
		return logits


class EFBook(nn.Module):
	def __init__(self, depth, query_size, key_size, value_size, dict_len, emb_size, num_heads, expansion, conv_dropout,
				 self_dropout, cross_dropout, cls_dropout, num_classes, device):
		super().__init__()
		# embedding size and dropout rate
		self.temporal_conv_layer = TemporalConvLayer(emb_size, conv_dropout)
 
		with torch.no_grad():
			eeg, nirs = torch.randn(1, 30, 4000), torch.randn(1, 36, 200)
			eeg_token, nirs_token = self.temporal_conv_layer(eeg, nirs)
			channels = [eeg_token.shape[-1], nirs_token.shape[-1]]

		print('Stage I: fine-grained representations')
		self.transformer = Transformer(depth, query_size, key_size, value_size, emb_size, num_heads, channels,
									   expansion, device, self_dropout, cross_dropout)
		
		print('Stage II: vector quantization')
		self.quantizer = Quantization(dict_len, emb_size)

		print('Stage III: feature aggregation & classification')
		self.classifier = Classifier(emb_size, num_classes)

	def forward(self, eeg, nirs):
		temporal_eeg, temporal_nirs = self.temporal_conv(eeg, nirs)
		temporal_eeg = temporal_eeg.squeeze(-2).permute(0,2,1)
		temporal_nirs = temporal_nirs.squeeze(-2).permute(0,2,1)

		eeg_token, nirs_token = self.transformer(temporal_eeg, temporal_nirs)

		quant_eeg, q_loss_eeg = self.quantizer(eeg_token)
		quant_nirs, q_loss_nirs = self.quantizer(nirs_token)

		logits = self.classifier(
			eeg_token, nirs_token,
			quant_eeg, quant_nirs
		)

		return {
			'logits': logits,
			'quant_loss': q_loss_eeg + q_loss_nirs
		}
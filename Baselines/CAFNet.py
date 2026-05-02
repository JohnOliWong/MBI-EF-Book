import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
	def __init__(self, dim, mid_dim, dropout):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(dim, mid_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(mid_dim, dim),
			nn.ReLU()
		)
		self.norm = nn.LayerNorm(dim)

	def forward(self, x):
		x = self.net(x)
		x = self.norm(x)
		return x


class MCLSTM(nn.Module):
	def __init__(self, input_size1, input_size2, hidden_size, num_layers, dropout):
		super(MCLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.linear1 = nn.Linear(input_size1, hidden_size)
		self.linear2 = nn.Linear(input_size2, hidden_size)
		self.shared_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, eeg, nirs):
		eeg = self.linear1(eeg)
		nirs = self.linear2(nirs)

		h = torch.zeros((self.num_layers, eeg.size(0), self.hidden_size), dtype=eeg.dtype, device=eeg.device) # [num_layers, batch_size, hidden_size]
		c_e = torch.zeros((self.num_layers, eeg.size(0), self.hidden_size), dtype=eeg.dtype, device=eeg.device)
		c_f = torch.zeros((self.num_layers, nirs.size(0), self.hidden_size), dtype=nirs.dtype, device=nirs.device)

		eeg, _ = self.shared_lstm(eeg, (h, c_e))
		nirs, _ = self.shared_lstm(nirs, (h, c_f))

		eeg = self.dropout(eeg)
		nirs = self.dropout(nirs)
		return eeg, nirs


class Attention(nn.Module):
	def __init__(self, dim, heads, dim_head):
		super(Attention, self).__init__()
		self.to_Q = nn.Linear(dim, heads * dim_head, bias=False)
		self.to_K = nn.Linear(dim, heads * dim_head, bias=False)
		self.to_V = nn.Linear(dim, heads * dim_head, bias=False)
		self.norm = nn.LayerNorm(heads * dim_head)

	def attention(self, Q, K, V):
		d_k = K.size(-1)
		scores = torch.matmul(Q, K.transpose(1,2)) / math.sqrt(d_k)
		alpha_n = F.softmax(scores, dim=-1)
		output = torch.matmul(alpha_n, V)
		output = output.sum(1)
		return output, alpha_n

	def forward(self, x):
		Q = self.to_Q(x)
		K = self.to_K(x)
		V = self.to_V(x)
		out, _ = self.attention(Q, K, V)
		out = self.norm(out)
		return out


class LinearLayer(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))

	def forward(self, x):
		x = self.clf(x)
		return x


class EEGSubNet(nn.Module):
	def __init__(self, dropout, dim, heads, dim_head, mlp_dim):
		super(EEGSubNet, self).__init__()
		self.SelfAttention = Attention(dim, heads, dim_head)
		self.FeedForward = FeedForward(heads * dim_head, mlp_dim, dropout)
	
	def forward(self, x):
		x = self.SelfAttention(x)
		x = self.FeedForward(x)
		return x


class NIRSSubNet(nn.Module):
	def __init__(self, dropout, dim, heads, dim_head, mlp_dim):
		super(NIRSSubNet, self).__init__()
		self.SelfAttention = Attention(dim, heads, dim_head)
		self.FeedForward = FeedForward(heads * dim_head, mlp_dim, dropout)
	
	def forward(self, x):
		x = self.SelfAttention(x)
		x = self.FeedForward(x)
		return x


class RegressionSubNetwork(nn.Module):
	def __init__(self, dim):
		super(RegressionSubNetwork, self).__init__()
		self.layers = nn.ModuleList([LinearLayer(dim, 1)])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x


class ClassificationSubNetwork(nn.Module):
	def __init__(self, dim, num_classes):
		super(ClassificationSubNetwork, self).__init__()
		self.layers = nn.ModuleList([LinearLayer(dim, num_classes)])

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x


class CAFNet(nn.Module):
	def __init__(self, eeg_dim, nirs_dim, hidden_size, num_layers, dim, heads, dim_head, mlp_dim, num_classes, dropout):
		super().__init__()
		self.num_classes = num_classes
		self.mc_lstm = MCLSTM(eeg_dim, nirs_dim, hidden_size, num_layers, dropout)
		self.eeg_subnet = EEGSubNet(dropout, dim, heads, dim_head, mlp_dim)
		self.nirs_subnet = NIRSSubNet(dropout, dim, heads, dim_head, mlp_dim)
		self.Regression = RegressionSubNetwork(heads * dim_head)
		self.Classification = ClassificationSubNetwork(heads * dim_head, num_classes)
		self.FeedForward = FeedForward(2 * heads * dim_head, mlp_dim, dropout)
		self.fc = nn.Linear(2 * heads * dim_head, num_classes)
		
	def confidence_loss(self, TCPLogit, TCPConfidence, label):
		pred = F.softmax(TCPLogit, dim=1)
		p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1).type(torch.int64)).view(-1)
		c_loss = torch.mean(F.mse_loss(TCPConfidence.view(-1), p_target, reduction='none'))
		return c_loss

	def KD_loss(self, TCPLogit_eeg, TCPLogit_nirs):
		loss1 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(TCPLogit_eeg, dim=1), F.softmax(TCPLogit_nirs, dim=1))
		loss2 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(TCPLogit_nirs, dim=1), F.softmax(TCPLogit_eeg, dim=1))
		return (loss1 + loss2) / 2
	
	def forward(self, eeg, nirs, label):
		# shared LSTM
		eeg, nirs = self.mc_lstm(eeg, nirs)

		# self-attention
		eeg = self.eeg_subnet(eeg)
		nirs = self.nirs_subnet(nirs)
		
		# regression model to return the modality confidence
		TCPConfidence_eeg = self.Regression(eeg)
		TCPConfidence_nirs = self.Regression(nirs)

		# un-normalized classification logits
		TCPLogit_eeg = self.Classification(eeg)
		TCPLogit_nirs = self.Classification(nirs)

		eeg = eeg * TCPConfidence_eeg
		nirs = nirs * TCPConfidence_nirs
		
		feature = torch.cat([eeg, nirs], dim=1)
		feature = self.FeedForward(feature) # extract and augment key features and introduce nonlinearity
		Logit = self.fc(feature)
		
		# confidence loss
		c_loss_eeg = self.confidence_loss(TCPLogit_eeg, TCPConfidence_eeg, label)
		c_loss_nirs = self.confidence_loss(TCPLogit_nirs, TCPConfidence_nirs, label)
		c_loss = c_loss_eeg + c_loss_nirs

		# kl-divergence loss
		kd_loss = self.KD_loss(TCPLogit_eeg, TCPLogit_nirs)
		
		return Logit, c_loss, kd_loss
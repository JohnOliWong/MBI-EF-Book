import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Dataloader.Dataloader_Excel import read_excel_eeg, read_excel_nirs
from TemporalTransformer import FGTransformer as fg

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import KFold
import pandas as pd

# Hyperparameters
depth = 4
depth_series = [4, 4] # dedicated for TSMMF
query_size = 64
key_size = 64
value_size = 64
emb_size = 64
dict_len = 512
num_heads = 4
expansion = 2
conv_dropout = 0.3
self_dropout = 0.3
cross_dropout = 0.3
cls_dropout = 0.5
num_classes = 2 # number of paradigm
batch_size = 16
num_epochs = 200
learning_rate = 0.001
mode = 0 # 0 = MI, 1 = MA

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories for results
os.makedirs("Results", exist_ok=True)

# define model
model = fg(
depth, query_size, key_size, value_size, emb_size, num_heads, expansion,
conv_dropout, self_dropout, cross_dropout, cls_dropout, num_classes, device).to(device)

# model was converted to torch.float64 to avoid overflow ealier
model = model.to(torch.float64)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for subject in range(29):
	subject += 1
	eeg, labels = read_excel_eeg(subject, mode)
	nirs, _ = read_excel_nirs(subject, mode)
	eeg, nirs, labels = eeg.to(device), nirs.to(device), labels.to(device)

	train_size = int(0.6 * len(eeg))
	eval_size = len(eeg) - train_size

	# create a tensor dataset
	dataset = torch.utils.data.TensorDataset(eeg, nirs, labels)

	# randomly split the data into training and evaluation sets
	train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

	acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], []

	# define subject-wise codebook
	m_v = torch.randn(dict_len, emb_size)
	codebook = m_v.clone()

	# Training loop
	for epoch in range(num_epochs):
		model.train()
		total_correct, total_loss = 0, 0

		for batch_eeg, batch_nirs, batch_labels in train_loader:
			batch_eeg = batch_eeg.to(device, dtype=torch.float64)
			batch_nirs = batch_nirs.to(device, dtype=torch.float64)
			batch_labels = batch_labels.to(device)

			optimizer.zero_grad()
			eeg_tokens, nirs_tokens = model(batch_eeg, batch_nirs) # (16, 201, 64), (16, 51, 64)
			print('Stage I completed')

			batch_tokens = torch.cat([eeg_tokens, nirs_tokens], dim=1) # [16, 64]
			quan_dis = torch.cdist(batch_tokens, codebook.weight)
			quan_index = torch.argmin(quan_dis, dim=-1)
			quan_tokens = codebook(quan_index)

			quan_eeg = quan_tokens[:,:201,:]
			quan_nirs = quan_tokens[:,201:,:]
			print('Stage II completed')
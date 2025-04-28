from Dataloader.Dataloader_Excel import read_excel_eeg, read_excel_nirs
from EFBook import EFBook as ef

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import KFold
import pandas as pd
import random
from collections import defaultdict

class Trainer:
	def __init__(self, config):
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.model = ef(
			config['depth'],
			config['query_size'],
			config['key_size'],
			config['value_size'],
			config['dict_len'],
			config['emb_size'],
			config['num_heads'],
			config['expansion'],
			config['conv_dropout'],
			config['self_dropout'],
			config['cross_dropout'],
			config['cls_dropout'],
			config['num_classes'],
			self.device,
		).to(self.device).to(torch.float64)

		self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
		self.criterion = nn.CrossEntropyLoss()

		self.init_codebook(config['dict_len'], config['emb_size'])

		os.makedirs('Results', exist_ok=True)

	def init_codebook(self, dict_len, emb_size):
		self.codebook = torch.randn(1, dict_len, emb_size).to(self.device)
		self.m = self.codebook.clone()
		self.N = torch.ones(dict_len).to(self.device)
		self.status = torch.zeros(dict_len).to(self.device)
		self.decay = 0.99
		self.step = 0
		return

	def codebook_update(self, H):
		dict_len = self.codebook.shape[1]
		for v in range(dict_len):
			if H[v] > 0:
				self.N[v] = self.decay * self.N[v] + (1 - self.decay) * len(H[v])
				self.m[v] = self.decay * self.m[v] + (1 - self.decay) * torch.sum(torch.stack(H[v]), dim=0)
				self.codebook[0, v] = self.N[v] / self.m[v]
			else:
				self.status[v] += 1
				if self.status[v] >= 100:
					active_v = [v for v in range(dict_len) if H[v] > 0]
					if active_v: # only proceed if activated codewords exist
						self.status[v] = 0
						random_v = random.choice(active_v)
						self.codebook[0, v] = self.codebook[0, random_v]
						self.N[v] = 1
						self.m[v] = self.codebook[0, v]
	
	def train_epoch(self, train_loader):
		self.model.train()
		total_correct, total_loss = 0, 0

		for batch_eeg, batch_nirs, batch_labels in train_loader:
			batch_eeg = batch_eeg.to(self.device, dtype=torch.float64)
			batch_nirs = batch_nirs.to(self.device, dtype=torch.float64)
			batch_labels = batch_labels.to(self.device)

			self.optimizer.zero_grad()
			
			# n_trial = eeg_token.shape[0]
			# for i in range(n_trial):
			# 	self.step += 1
			# 	eeg_slice, nirs_slice = eeg_token[i, :, :], nirs_token[i, :, :]
			# 	vector = torch.cat([eeg_slice, nirs_slice], dim=0)  ###### Changed dim from 1 to 0
			# 	vector_len = vector.shape[0]
				
			# 	# Initialize H as defaultdict of lists
			# 	H = defaultdict(list)
			# 	quan_idx = []
				
			# 	for j in range(vector_len):
			# 		distances = torch.norm(vector[j, :] - self.codebook[0], p=2, dim=1)
			# 		v = torch.argmin(distances).item()
			# 		quan_idx.append(v)
			# 		H[v].append(vector[j, :])
				
			# 	# Update codebook
			# 	self.codebook_update(H)
				
			# 	# Quantize the tokens
			# 	quan_eeg_slice = self.codebook[:, quan_idx[:eeg_slice.shape[0]], :]
			# 	quan_nirs_slice = self.codebook[:, quan_idx[eeg_slice.shape[0]:], :]
			
			outputs = self.model(batch_eeg, batch_nirs)
			loss = self.criterion(outputs, batch_labels)
			loss.backward()
			self.optimizer.step()
			
			# Compute accuracy
			preds = torch.argmax(outputs, dim=1)
			total_correct += (preds == batch_labels).sum().item()
			total_loss += loss.item()

			train_loss = total_loss / len(train_loader)
			train_acc = total_correct / len(train_loader.dataset)
			
			return train_loss, train_acc
	
	def evaluate_epoch(self, eval_loader):
		self.model.eval()
		total_loss, total_correct = 0, 0
		all_preds, all_labels = [], []
		
		with torch.no_grad():
			for eval_eeg, eval_nirs, eval_labels in eval_loader:
				eval_eeg = eval_eeg.to(self.device, dtype=torch.float64)
				eval_nirs = eval_nirs.to(self.device, dtype=torch.float64)
				eval_labels = eval_labels.to(self.device)
				
				outputs = self.model(eval_eeg, eval_nirs) # feature extraction, vector quantization, feature aggregation and classification should be done inside "model"
				loss = self.criterion(outputs, eval_labels)
				total_loss += loss.item()
				
				preds = torch.argmax(outputs, dim=1)
				total_correct += (preds == eval_labels).sum().item()
				
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(eval_labels.cpu().numpy())
		
		loss = total_loss / len(eval_loader)
		acc = total_correct / len(eval_loader.dataset)
		precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
		kappa = cohen_kappa_score(all_labels, all_preds)
		
		print(f"Evaluation Loss: {loss:.2f} | Evaluation Acc: {acc:.2f}")
		return acc, precision, recall, f1, kappa

	def train_subject(self, subject_id, mode):
		eeg, labels = read_excel_eeg(subject_id, mode)
		nirs, _ = read_excel_nirs(subject_id, mode)
		eeg, nirs, labels = eeg.to(self.device), nirs.to(self.device), labels.to(self.device)
		
		train_size = int(0.6 * len(eeg)) # 60/40 for training/testing
		eval_size = len(eeg) - train_size
		
		dataset = torch.utils.data.TensorDataset(eeg, nirs, labels)
		train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
		eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.config['batch_size'], shuffle=False)
		
		acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], []
		
		# Reset codebook variables for new subject
		self.init_codebook_vars(self.config['dict_len'], self.config['emb_size'])
		
		for epoch in range(self.config['num_epochs']):
			train_loss, train_acc = self.train_epoch(train_loader)
			eval_acc, precision, recall, f1, kappa = self.evaluate_epoch(eval_loader)
			
			acc_list.append(eval_acc)
			precision_list.append(precision)
			recall_list.append(recall)
			f1_list.append(f1)
			kappa_list.append(kappa)
			
			print(f"Subject {subject_id} | Epoch {epoch+1}/{self.config['num_epochs']} | "
				f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
				f"Eval Acc: {eval_acc:.4f}")

		mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)
		mean_precision, std_precision = np.mean(precision_list), np.std(precision_list)
		mean_recall, std_recall = np.mean(recall_list), np.std(recall_list)
		mean_f1, std_f1 = np.mean(f1_list), np.std(f1_list)
		mean_kappa = np.mean(kappa_list)

		mean_acc, std_acc = mean_acc*100, std_acc*100
		mean_precision, std_precision = mean_precision*100, std_precision*100
		mean_recall, std_recall = mean_recall*100, std_recall*100
		mean_f1, std_f1 = mean_f1*100, std_f1*100

		log_path = f'Results/log_s{subject_id:2d}.txt'
		with open(log_path, 'a') as log_file:
			log_file.write(f'Accuracy: {mean_acc:.2f} ± {std_acc:.2f}\n')
			log_file.write(f'Precision: {mean_precision:.2f} ± {std_precision:.2f}\n')
			log_file.write(f'Recall: {mean_recall:.2f} ± {std_recall:.2f}\n')
			log_file.write(f'F1: {mean_f1:.2f} ± {std_f1:2f}\n')
			log_file.write(f'Kappa: {mean_kappa:.2f}\n')
			log_file.write('\n')

		log_excel = 'Results/Log.xlsx'
		new_row = pd.DataFrame([[mean_acc, std_acc, mean_precision, std_precision, mean_recall, std_recall, mean_f1, std_f1, mean_kappa]], 
							columns=['Accuracy', 'Std_Accuracy', 'Precision', 'Std_Precision', 'Recall', 'Std_Recall', 'F1', 'Std_F1', 'Kappa'])
		new_row = new_row.round(2)
		if not os.path.exists(log_excel):
			new_row.to_excel(log_excel, index=False, engine='openpyxl')
		else:
			with pd.ExcelWriter(log_excel, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
				if 'Sheet1' in writer.sheets:
					startrow = writer.sheets['Sheet1'].max_row
				else:
					startrow = 0
				new_row.to_excel(writer, index=False, header=False, startrow=startrow)

		print(f'Subject:{subject_id} Accuracy: {mean_acc:.2f} ± {std_acc:.2f}\n')			
		return

# Hyperparameters
config = {
	'depth': 4,
	'query_size': 64,
	'key_size': 64,
	'value_size': 64,
	'emb_size': 64,
	'dict_len': 512,
	'num_heads': 4,
	'expansion': 2,
	'conv_dropout': 0.3,
	'self_dropout': 0.3,
	'cross_dropout': 0.3,
	'cls_dropout': 0.5,
	'num_classes': 2,
	'batch_size': 16,
	'num_epochs': 200,
	'learning_rate': 0.001
}

# Initialize and run trainer
trainer = Trainer(config)
for subject in range(29):
	subject += 1
	print(f"\n=== Subject {subject} ===")
	results = trainer.train_subject(subject, mode=0) # 0 = MI, 1 = MA
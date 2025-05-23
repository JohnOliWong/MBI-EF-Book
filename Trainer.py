from Dataloader.Dataloader_Excel import read_excel_eeg, read_excel_nirs
from EFBook_Hierarchical import EFBook as ef

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
			config['decay'],
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

		os.makedirs('Results', exist_ok=True)
	
	def train_epoch(self, train_loader):
		self.model.train()
		total_correct, total_loss = 0, 0

		for batch_eeg, batch_nirs, batch_labels in train_loader:
			batch_eeg = batch_eeg.to(self.device, dtype=torch.float64)
			batch_nirs = batch_nirs.to(self.device, dtype=torch.float64)
			batch_labels = batch_labels.to(self.device)
			
			self.optimizer.zero_grad()
			model_output = self.model(batch_eeg, batch_nirs)
			outputs = model_output['outputs']
			quan_loss = model_output['quan_loss']

			quan_lambda = 0.1
			cls_loss = self.criterion(outputs, batch_labels)
			loss = cls_loss + quan_loss * quan_lambda
			loss.backward()
			self.optimizer.step()
			
			preds = torch.argmax(outputs, dim=1)
			total_correct += (preds == batch_labels).sum().item()
			total_loss += loss.item()

			train_loss = total_loss / len(train_loader)
			train_acc = total_correct / len(train_loader.dataset)
			print(f"Training Loss: {train_loss:.2f} | Training Acc: {train_acc:.2f}")
			
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
				
				model_output = self.model(eval_eeg, eval_nirs)
				outputs = model_output['outputs']
				quan_loss = model_output['quan_loss']
				
				quan_lambda = 0.1
				cls_loss = self.criterion(outputs, eval_labels)
				loss = cls_loss + quan_loss * quan_lambda
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
		
		train_size = int(0.6 * len(eeg)) # 60 trials/subject, 60/40 for training/testing
		eval_size = len(eeg) - train_size
		
		dataset = torch.utils.data.TensorDataset(eeg, nirs, labels)
		train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
		eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.config['batch_size'], shuffle=False)
		
		acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], []
		
		for epoch in range(self.config['num_epochs']):
			train_loss, train_acc = self.train_epoch(train_loader)
			eval_acc, precision, recall, f1, kappa = self.evaluate_epoch(eval_loader)
			
			acc_list.append(eval_acc)
			precision_list.append(precision)
			recall_list.append(recall)
			f1_list.append(f1)
			kappa_list.append(kappa)
			
			print(f"Subject {subject_id} | Epoch {epoch+1}/{self.config['num_epochs']} | "
				f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f} | "
				f"Eval Acc: {eval_acc:.2f}")

		mean_acc, std_acc = np.mean(acc_list[-50:]), np.std(acc_list[-50:])
		mean_precision, std_precision = np.mean(precision_list[-50:]), np.std(precision_list[-50:])
		mean_recall, std_recall = np.mean(recall_list[-50:]), np.std(recall_list[-50:])
		mean_f1, std_f1 = np.mean(f1_list[-50:]), np.std(f1_list[-50:])
		mean_kappa = np.mean(kappa_list[-50:])

		mean_acc, std_acc = mean_acc * 100, std_acc * 100
		mean_precision, std_precision = mean_precision * 100, std_precision * 100
		mean_recall, std_recall = mean_recall * 100, std_recall * 100
		mean_f1, std_f1 = mean_f1 * 100, std_f1 * 100

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
	'decay': 0.99,
	'num_heads': 4,
	'expansion': 2,
	'conv_dropout': 0.3,
	'self_dropout': 0.3,
	'cross_dropout': 0.3,
	'cls_dropout': 0.5,
	'num_classes': 2,
	'batch_size': 16,
	'num_epochs': 5,
	'learning_rate': 1e-3
}

# Initialize and run trainer
trainer = Trainer(config)
for subject in range(29):
	subject += 1
	print(f"\n=== Subject {subject} ===")
	results = trainer.train_subject(subject, mode=0) # 0 = MI, 1 = MA
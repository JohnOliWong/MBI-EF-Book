from EFBook_DWConv_PS_V4 import EFBook as ef
from Metrics import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import os
import shutil
import pickle

torch.manual_seed(42)
torch.cuda.manual_seed(42)

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
			config['mode'],
			self.device,
		).to(self.device).to(torch.float64)

		self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
		self.criterion = nn.CrossEntropyLoss()

		os.makedirs('Results', exist_ok=True)
	
	def z_score_mm(self, train_dataset, test_dataset, eps=1e-6):
		train_eeg = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
		train_eeg_mean = train_eeg.mean(dim=(0,2), keepdim=True)
		train_eeg_std = train_eeg.std(dim=(0,2),keepdim=True)
		train_eeg = (train_eeg - train_eeg_mean) / (train_eeg_std + eps)

		# print(train_eeg_std.mean().item())
		# print(train_eeg_std.std().item())

		train_nirs = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
		train_nirs_mean = train_nirs.mean(dim=(0,2), keepdim=True)
		train_nirs_std = train_nirs.std(dim=(0,2),keepdim=True)
		train_nirs = (train_nirs - train_nirs_mean) / (train_nirs_std + eps)

		# print(train_nirs_std.mean().item())
		# print(train_nirs_std.std().item())

		train_labels = torch.stack([train_dataset[i][2] for i in range(len(train_dataset))])
		train_dataset = torch.utils.data.TensorDataset(train_eeg, train_nirs, train_labels)

		eval_eeg = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
		eval_eeg = (eval_eeg - train_eeg_mean) / (train_eeg_std + eps)
		eval_nirs = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
		eval_nirs = (eval_nirs - train_nirs_mean) / (train_nirs_std + eps)
		eval_labels = torch.stack([test_dataset[i][2] for i in range(len(test_dataset))])
		test_dataset = torch.utils.data.TensorDataset(eval_eeg, eval_nirs, eval_labels)

		return train_dataset, test_dataset
	
	def train_epoch(self, epoch, train_loader):
		epoch += 1
		self.model.train()
		total_correct, total_loss = 0, 0

		for batch_eeg, batch_nirs, batch_labels in train_loader:
			batch_eeg = batch_eeg.to(self.device, dtype=torch.float64)
			batch_nirs = batch_nirs.to(self.device, dtype=torch.float64)
			batch_labels = batch_labels.to(self.device)
			
			model_output = self.model(batch_eeg, batch_nirs)
			outputs = model_output['outputs']
			quan_loss = model_output['quan_loss']

			quan_lambda = config['quan_lambda']
			cls_loss = self.criterion(outputs, batch_labels)
			loss = cls_loss + quan_loss * quan_lambda
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			
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
				
				model_output = self.model(eval_eeg, eval_nirs)
				outputs = model_output['outputs']
				quan_loss = model_output['quan_loss']
				
				quan_lambda = config['quan_lambda']
				cls_loss = self.criterion(outputs, eval_labels)
				loss = cls_loss + quan_loss * quan_lambda
				total_loss += loss.item()
				
				preds = torch.argmax(outputs, dim=1)
				total_correct += (preds == eval_labels).sum().item()
				
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(eval_labels.cpu().numpy())
		
		loss = total_loss / len(eval_loader)
		acc = total_correct / len(eval_loader.dataset)
		precision = precision_score(all_labels, all_preds, zero_division=0)
		recall = recall_score(all_labels, all_preds)
		f1 = f1_score(all_labels, all_preds)
		kappa = cohen_kappa_score(all_labels, all_preds)
		
		return loss, acc, precision, recall, f1, kappa

	def train_subject(self, subject, mode):
		if mode == 0:
			data_root = config['mi_root'] + str(subject) + '.pkl'
		elif mode == 1:
			data_root = config['ma_root'] + str(subject) + '.pkl'
		elif mode == 2:
			data_root = config['wg_root'] + str(subject) + '.pkl'
		with open(data_root, 'rb') as f:
			data = pickle.load(f)
		
		eeg = data['eeg']
		nirs = data['nirs']
		labels = data['labels']
		
		if mode == 0 or mode == 1:
			eeg = eeg.unsqueeze(1)
			nirs = nirs.unsqueeze(1)
		elif mode == 2:
			eeg = torch.tensor(eeg, dtype=torch.float64)
			nirs = torch.tensor(nirs, dtype=torch.float64)
		eeg, nirs, labels = eeg.to(self.device), nirs.to(self.device), labels.to(self.device)
		
		train_size = int(config['ratio'] * len(eeg)) # training/testing ratio
		eval_size = len(eeg) - train_size
		
		dataset = torch.utils.data.TensorDataset(eeg, nirs, labels)
		train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
		if config['z_score']:
			train_dataset, eval_dataset = self.z_score_mm(train_dataset, eval_dataset)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
		eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.config['batch_size'], shuffle=False)
		
		acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], []
		
		for epoch in range(self.config['num_epochs']):
			train_loss, train_acc = self.train_epoch(epoch, train_loader)
			eval_loss, eval_acc, precision, recall, f1, kappa = self.evaluate_epoch(eval_loader)
			
			acc_list.append(eval_acc)
			precision_list.append(precision)
			recall_list.append(recall)
			f1_list.append(f1)
			kappa_list.append(kappa)
			
			print(f"Subject {subject} | Epoch {epoch+1}/{self.config['num_epochs']} | "
				f"Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f} | "
				f"Eval Loss: {eval_loss:.2f} | Eval Acc: {eval_acc:.2f}")
			
		metrics(subject, config['log_name'], config['log_mode'], acc_list, precision_list, recall_list, f1_list, kappa_list)
		return

# Hyperparameters
# the correlation between Q, K, V and embedding size
config = {
	'depth': 4,
	'query_size': 128,
	'key_size': 128,
	'value_size': 128,
	'emb_size': 128,
	'dict_len': 64,
	'decay': 0.99,
	'num_heads': 4,
	'expansion': 2,
	'conv_dropout': 0.3,
	'self_dropout': 0.3,
	'cross_dropout': 0.3,
	'cls_dropout': 0.5,
	'num_classes': 2,
	'batch_size': 16,
	'num_epochs': 200,
	'learning_rate': 1e-3,
	'ratio': 0.6,
	'mode': 0,
	'log_name': '805',
	'log_mode': 1,
	'z_score': True,
	'quan_lambda': 0.1,
	'mi_root': '../../Dataset/EF-MI-MA/EF-PKL-MI/',
	'ma_root': '../../Dataset/EF-MI-MA/EF-PKL-MA/',
	'wg_root': '../../Dataset/EF-WG/WG/',
}

# Initialize and run trainer
current_dir = os.getcwd()
results_dir = os.path.join(current_dir, 'Results', config['log_name'])
file_name = 'SD.ipynb'

source_dir = os.path.join(current_dir, file_name)
destination_dir = results_dir

for run in range(1):
	trainer = Trainer(config)
	mode = config['mode']
	num_subject = (29 if mode == 0 or mode == 1 else 26) # MI = 0, MA = 1, WG = 2
	for subject in range(num_subject):
		subject += 1
		print(f"\n=== Subject {subject} ===")
		results = trainer.train_subject(subject, mode)

os.makedirs(destination_dir, exist_ok=True)
shutil.copy2(source_dir, destination_dir)
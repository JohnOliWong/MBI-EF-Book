from EF_Book_VQ import EF_Book as ef
from EF_Book_SharedVQ import EF_Book as ef_sharedvq
from EF_Book_PrivateVQ import EF_Book as ef_privatevq
from EF_Book_NoVQ import EF_Book as ef_novq
from EF_Book_EEG import EF_Book as ef_eeg
from EF_Book_NIRS import EF_Book as ef_nirs
from Dataloader_EF import read_ef_train_si as si_data
from Metrics import metrics, load_param

import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import os
import re
from pathlib import Path


class Trainer:
	def __init__(self, config):
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		if config['model_id'] == 0:
			self.model = ef(config['dict_len'], config['emb_size'], config['num_class'], config['threshold'], self.device).to(self.device).to(torch.float64)
		elif config['model_id'] == 1:
			self.model = ef_sharedvq(config['dict_len'], config['emb_size'], config['num_class'], config['threshold'], self.device).to(self.device).to(torch.float64)
		elif config['model_id'] == 2:
			self.model = ef_privatevq(config['dict_len'], config['emb_size'], config['num_class'], config['threshold'], self.device).to(self.device).to(torch.float64)
		elif config['model_id'] == 3:
			self.model = ef_novq(config['dict_len'], config['emb_size'], config['num_class'], config['threshold'], self.device).to(self.device).to(torch.float64)
		elif config['model_id'] == 4:
			self.model = ef_eeg(config['dict_len'], config['emb_size'], config['num_class'], config['threshold'], self.device).to(self.device).to(torch.float64)
		elif config['model_id'] == 5:
			self.model = ef_nirs(config['dict_len'], config['emb_size'], config['num_class'], config['threshold'], self.device).to(self.device).to(torch.float64)

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['learning_rate'])
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.optimizer,
			mode='min',
			factor=0.8,
			patience=5,
			min_lr=0.1*config['learning_rate'],
		)

		os.makedirs('Results', exist_ok=True)
	
	def z_score_mm(self, train_dataset, test_dataset, eps=1e-6):
		train_eeg = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
		train_eeg_mean = train_eeg.mean(dim=(0,1), keepdim=True)
		train_eeg_std = train_eeg.std(dim=(0,1),keepdim=True)
		train_eeg = (train_eeg - train_eeg_mean) / (train_eeg_std + eps)

		train_nirs = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
		train_nirs_mean = train_nirs.mean(dim=(0,1), keepdim=True)
		train_nirs_std = train_nirs.std(dim=(0,1),keepdim=True)
		train_nirs = (train_nirs - train_nirs_mean) / (train_nirs_std + eps)

		train_labels = torch.stack([train_dataset[i][2] for i in range(len(train_dataset))])
		train_dataset = torch.utils.data.TensorDataset(train_eeg, train_nirs, train_labels)

		eval_eeg = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
		eval_eeg = (eval_eeg - train_eeg_mean) / (train_eeg_std + eps)
		eval_nirs = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
		eval_nirs = (eval_nirs - train_nirs_mean) / (train_nirs_std + eps)
		eval_labels = torch.stack([test_dataset[i][2] for i in range(len(test_dataset))])
		test_dataset = torch.utils.data.TensorDataset(eval_eeg, eval_nirs, eval_labels)

		return train_dataset, test_dataset
	
	def evaluate_epoch(self, results_root, subject, eval_loader):
		self.model.load_state_dict(torch.load(results_root + f'{subject}.pt'))
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
				preds = torch.argmax(outputs, dim=1)
				total_correct += (preds == eval_labels).sum().item()
				
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(eval_labels.cpu().numpy())
		
		acc = total_correct / len(eval_loader.dataset)
		precision = precision_score(all_labels, all_preds, zero_division=0)
		recall = recall_score(all_labels, all_preds, zero_division=0)
		f1 = f1_score(all_labels, all_preds, zero_division=0)
		kappa = cohen_kappa_score(all_labels, all_preds)
		
		return acc, precision, recall, f1, kappa
	
	def test_subject(self, results_root, subject, mode):
		train_dataset, eval_dataset = si_data(subject, mode)
		if config['z_score']:
			train_dataset, eval_dataset = self.z_score_mm(train_dataset, eval_dataset)
		eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.config['batch_size'], shuffle=False)
		eval_acc, precision, recall, f1, kappa = self.evaluate_epoch(results_root, subject, eval_loader)
		print(f"Subject {subject} | Eval Acc: {eval_acc:.2f}")

		return eval_acc, precision, recall, f1, kappa

# Load parameters
exp_name = '8014/'
results_root = 'Results/' + exp_name
config, seeds = load_param(results_root)
results_path = Path(results_root)
subject_list = sorted(
	int(re.search(r'^(\d+)\.pt$', file.name).group(1))
	for file in results_path.glob('*.pt')
	if re.search(r'^(\d+)\.pt$', file.name)
)

acc_list, precision_list, recall_list, f1_list, kappa_list = [], [], [], [], []
for i, subject in enumerate(subject_list):
	seed = seeds[i]
	# seed = random.randint(1, 2025)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	print(f"\n=== Subject {subject} ===")
	print(f'Seed is {seed}')
	trainer = Trainer(config)
	eval_acc, precision, recall, f1, kappa = trainer.test_subject(results_root, subject, config['mode'])
	acc_list.append(eval_acc)
	precision_list.append(precision)
	recall_list.append(recall)
	f1_list.append(f1)
	kappa_list.append(kappa)

acc_list = [acc * 100 for acc in acc_list]
precision_list = [precision * 100 for precision in precision_list]
recall_list = [recall * 100 for recall in recall_list]
f1_list = [f1 * 100 for f1 in f1_list]
data = {
	'Acc': acc_list,
	'Precision': precision_list,
	'Recall': recall_list,
	'F1': f1_list,
	'Kappa': kappa_list,
	}
exp_name = results_root.split('/')[-2]
log_root = results_root + f'{exp_name}.xlsx'
metrics(log_root, data)
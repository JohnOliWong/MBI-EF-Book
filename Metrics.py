import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
import os


def train_metrics(log_root, log_name, acc_list, precision_list, recall_list, f1_list, kappa_list):
	if not os.path.exists(log_root):
		os.makedirs(log_root)
	log_excel = log_root + f'{log_name}.xlsx'
	acc_list = [acc * 100 for acc in acc_list]
	precision_list = [precision * 100 for precision in precision_list]
	recall_list = [recall * 100 for recall in recall_list]
	f1_list = [f1 * 100 for f1 in f1_list]
	data = {
		'Accuracy': acc_list,
		'Precision': precision_list,
		'Recall': recall_list,
		'F1': f1_list,
		'Kappa': kappa_list
	}
	new_row = pd.DataFrame(data).round(2)
	if not os.path.exists(log_excel):
		new_row.to_excel(log_excel, index=False, engine='openpyxl')
	else:
		with pd.ExcelWriter(log_excel, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
			if 'Sheet1' in writer.sheets:
				startrow = writer.sheets['Sheet1'].max_row
			else:
				startrow = 0
			new_row.to_excel(writer, index=False, header=False, startrow=startrow)
	
	print('Logging Completed')

def eval_metrics(log_root, acc_list, precision_list, recall_list, f1_list, kappa_list):
	if not os.path.exists(log_root):
		os.makedirs(log_root)
	exp_name = log_root.split('/')[-2]
	log_excel = log_root + f'{exp_name}.xlsx'
	acc_list = [acc * 100 for acc in acc_list]
	precision_list = [precision * 100 for precision in precision_list]
	recall_list = [recall * 100 for recall in recall_list]
	f1_list = [f1 * 100 for f1 in f1_list]
	data = {
		'Accuracy': acc_list,
		'Precision': precision_list,
		'Recall': recall_list,
		'F1': f1_list,
		'Kappa': kappa_list
	}
	new_row = pd.DataFrame(data).round(2)
	if not os.path.exists(log_excel):
		new_row.to_excel(log_excel, index=False, engine='openpyxl')
	else:
		with pd.ExcelWriter(log_excel, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
			if 'Sheet1' in writer.sheets:
				startrow = writer.sheets['Sheet1'].max_row
			else:
				startrow = 0
			new_row.to_excel(writer, index=False, header=False, startrow=startrow)
	
	print('Logging Completed')

def save_seed(exp_name, mode, seeds):
	log_root = 'Results/' + 'Seed.txt'

	with open(log_root, 'w') as f:
		f.write(exp_name + '\n')
		f.write(str(mode) + '\n')
		for seed in seeds:
			f.write(str(seed) + '\n')

def load_seed():
	log_root = 'Results/' + 'Seed.txt'

	with open(log_root, 'r') as f:
		lines = f.readlines()
		exp_name = lines[0].strip()
		mode = int(lines[1].strip())
		seeds = [int(line.strip()) for line in lines[1:]]

	return exp_name, mode, seeds
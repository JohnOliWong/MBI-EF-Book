import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
import os


def metrics(log_root, data):
	new_row = pd.DataFrame(data).round(2)
	if not os.path.exists(log_root):
		new_row.to_excel(log_root, index=False, engine='openpyxl')
	else:
		with pd.ExcelWriter(log_root, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
			if 'Sheet1' in writer.sheets:
				startrow = writer.sheets['Sheet1'].max_row
			else:
				startrow = 0
			new_row.to_excel(writer, index=False, header=False, startrow=startrow)
	
	print('Logging Completed')

def save_seed(results_root, mode, seeds):
	log_root = 'Results/' + 'Seed.txt'

	with open(log_root, 'w') as f:
		f.write(results_root + '\n')
		f.write(str(mode) + '\n')
		for seed in seeds:
			f.write(str(seed) + '\n')

def load_seed():
	log_root = 'Results/' + 'Seed.txt'

	with open(log_root, 'r') as f:
		lines = f.readlines()
		results_root = lines[0].strip()
		mode = int(lines[1].strip())
		seeds = [int(line.strip()) for line in lines[2:]]

	return results_root, mode, seeds
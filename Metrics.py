import pandas as pd
import os
import pickle


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

def save_params(results_root, config, seeds):
	log_root = results_root + 'Params.pkl'

	with open(log_root, 'wb') as f:
		pickle.dump(results_root, f)
		pickle.dump(config, f)
		pickle.dump(seeds, f)

def load_seed(results_root):
	log_root = results_root + 'Params.pkl'

	with open(log_root, 'rb') as f:
		results_root = pickle.load(f)
		config = pickle.load(f)
		seeds = pickle.load(f)
	
	return config, seeds
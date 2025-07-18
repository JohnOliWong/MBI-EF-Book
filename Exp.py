import numpy as np
import torch
import os


exp_name = '8008/'
results_root = 'Results/' + exp_name
print(results_root)
exp_name = results_root.split('/')[-2]
print(exp_name)

attachments = ('nohup.out' if os.path.exists('nohup.out') else None)
print(attachments)
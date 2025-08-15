import numpy as np
import torch
import os
from pathlib import Path


# exp_name = '8008/'
# results_root = 'Results/' + exp_name
# print(results_root)
# exp_name = results_root.split('/')[-2]
# print(exp_name)

# attachments = ('nohup.out' if os.path.exists('nohup.out') else None)
# print(attachments)

data_root = 'D:/HIT/MBI/1.xlsx'
data_root = Path(data_root)
data_root = str(data_root)
data_root = data_root.split('\\')
print(data_root)
new_root = '/'.join(data_root[:-1]) + '/'
print(new_root)
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

# data_root = 'D:/HIT/MBI/1.xlsx'
# data_root = Path(data_root)
# data_root = str(data_root)
# data_root = data_root.split('\\')
# print(data_root)
# new_root = '/'.join(data_root[:-1]) + '/'
# print(new_root)

# def happy_sum(n):
# 	sum = 0
# 	while n:
# 		digit = n % 10
# 		sum += digit ** 2
# 		n //= 10
# 	return sum

# print(happy_sum(100))

# def is_happy(n):
# 	hash_map = {}
# 	while True:
# 		n = happy_sum(n)
# 		if n in hash_map:
# 			return False
# 		if n == 1:
# 			return True
# 		hash_map[n] = 1

# print(is_happy(19))

# def firstUniqChar(s: str) -> int:
# 	hash_map = {}
# 	for i, letter in enumerate(s):
# 		if letter in hash_map:
# 			hash_map[letter] = -1
# 		hash_map[letter] = i
# 	return min(hash_map.values())

# print(firstUniqChar('aabb'))

def hIndex(citations) -> int:
	new_cite = [cite for cite in citations if cite != 0]
	if len(new_cite) == 0:
		return 0
	new_cite.sort()
	for index in range(len(new_cite), 0, -1):
		flag = len([cite for cite in new_cite if cite >= index]) >= index
		if flag:
			return index
	return

nums = [2,-1,3,2,2]
nums.sort()
new_nums = set(nums)
print(hIndex([1, 3, 1]))
import numpy as np
import torch
import os
from pathlib import Path
from collections import defaultdict, Counter
import bisect
from itertools import accumulate
import datetime


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

# def hIndex(citations) -> int:
# 	new_cite = [cite for cite in citations if cite != 0]
# 	if len(new_cite) == 0:
# 		return 0
# 	new_cite.sort()
# 	for index in range(len(new_cite), 0, -1):
# 		flag = len([cite for cite in new_cite if cite >= index]) >= index
# 		if flag:
# 			return index
# 	return

# nums = [2,-1,3,2,2]
# nums.sort()
# new_nums = set(nums)
# print(hIndex([1, 3, 1]))

# tri = [[1], [1, 1], [1, 2, 1]]
# print(tri[2][1])

# print(int("-2"))
# operations = ["5","-2","4","C","D","9","+","+"]
# stack = []
# for char in operations:
# 	if char.isdigit():
# 		stack.append(int(char))
# 	elif char == 'C':
# 		stack.pop()
# 	elif char == 'D':
# 		stack.append(2 * stack[-1])
# 	elif char == '+':
# 		stack.append(stack[-1]+stack[-2])
# print(stack)
# print(sum(int(char) for char in stack))

# temperatures = [73,74,75,71,69,72,76,73]
# n = len(temperatures)
# stack = [0]
# res = [-1] * n
# for i in range(1, n):
# 	temp = temperatures[i]
# 	while stack and temp > temperatures[stack[-1]]:
# 		index = stack.pop()
# 		res[index] = i
# 	stack.append(i)
# print(res)

# x, y = 1, 4
# bits_x = bin(x)[2:]
# bits_y = bin(y)[2:]
# m, n = len(bits_x), len(bits_y)
# if n > m:
# 	bits_x, bits_y = bits_y, bits_x
# diff = len(bits_x) - len(bits_y)
# if diff != 0:
# 	bits_y = '0' * diff + bits_y
# bits_x, bits_y = list(bits_x), list(bits_y)
# count = 0
# for i in range(m):
# 	if bits_x[i] != bits_y[i]:
# 		count += 1
# print(count)

# nums = [100,4,200,1,3,2]
# nums = [0,3,7,2,5,8,4,6,0,1]
# nums.sort()
# # nums = list(set(nums))
# # nums.sort()
# maxlen = 0
# currentlen = 1
# for i in range(1, len(nums)):
# 	if nums[i] == nums[i-1] + 1:
# 		currentlen += 1
# 	elif nums[i] == nums[i-1]:
# 		continue
# 	else:
# 		maxlen = max(maxlen, currentlen)
# 		currentlen = 0
# print(maxlen)

# paragraph = 'Bob hit a ball, the hit BALL flew far after it was hit.'
# banned = ['hit']
# word = ''
# words = []
# for c in paragraph:
# 	if c.isalpha():
# 		word += c.lower()
# 	else:
# 		if word:
# 			words.append(word)
# 			word = ''
# if word:
# 	words.append(word)

# hash_map = {}
# for word in words:
# 	if word not in banned:
# 		hash_map[word] = hash_map.get(word, 0) + 1

# count = Counter(words).most_common()
# print(count)
# print(hash_map)

# for item in count:
# 	if item[0] not in banned:
# 		print(item[0])

# flowerbed = [1,0,0,0,1]
# n = 1
# count = 0
# interval = 0
# for num in flowerbed:
# 	if flowerbed == 0:
# 		interval += 1
# 	else:
# 		if interval != 0:
# 			count += (interval - 1) // 2
# 		interval = 0
# if interval != 0:
# 	count += (interval - 1) // 2
# print(count)

# nums = [1,1,1,2,2,3]
# k = 2
# hash_map = {}
# for num in nums:
# 	hash_map[num] = hash_map.get(num, 0) + 1

# keys = hash_map.keys()
# index_map = {}
# for key in keys:
# 	index_map[hash_map[key]] = key

# index_list = list(index_map.keys())
# index_list.sort()
# res = [0] * k
# for i in range(k):
# 	res[i] = index_map[index_list[i]]

# grid = [[1,3,1],[1,5,1],[4,2,1]]
# m, n = len(grid), len(grid[0])
# dp = [[0] * n for _ in range(m)]
# val = 0
# for j in range(n):
# 	val += grid[0][j]
# 	dp[0][j] = val
# val = 0
# for i in range(m):
# 	val += grid[i][0]
# 	dp[i][0] = val

# for i in range(1, m):
# 	for j in range(1, n):
# 		dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
# print(dp)

# matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
# target = 5
# m, n = len(matrix), len(matrix[0])

# def row_bisect(row):
# 	left, right = 0, n-1
# 	while left <= right:
# 		mid = left + (right - left) // 2
# 		if matrix[row][mid] < target:
# 			left += 1
# 		elif matrix[row][mid] > target:
# 			right -= 1
# 		else:
# 			return mid
# 	return left
# col = row_bisect(0)
# col -= 1
# print(col)

# def col_bisect(col):
# 	left, right = 0, m-1
# 	while left <= right:
# 		mid = left + (right - left) // 2
# 		if matrix[mid][col] < target:
# 			left += 1
# 		elif matrix[mid][col] > target:
# 			right -= 1
# 		else:
# 			return mid
# 	return left
# row = col_bisect(col)
# print(matrix[row][col] == target)

# s = 'bb'
# n = len(s)
# dp = [[0] * n for _ in range(n+1)]
# for i in range(n):
# 	dp[i][i] = 1

# res = 1
# ps = s[0]
# for i in range(1, n):
# 	for j in range(i):
# 		print(i, j)
# 		if s[i] == s[j]:
# 			dp[i][j] = dp[i-1][j+1] + 2
# 		else:
# 			dp[i][j] = dp[i-1][j+1]
# 		if dp[i][j] > res:
# 			res = dp[i][j]
# 			ps = s[j:i+1]
# print(dp)


# from collections import defaultdict

# n = 3
# buildings = [[1,2],[2,2],[3,2],[2,1],[2,3]]
# n = 5
# buildings = [[1,3],[3,2],[3,3],[3,5],[5,3]]
# xg = defaultdict(list)
# yg = defaultdict(list)

# for b in buildings:
# 	x, y = b
# 	xg[x].append(b)
# 	yg[y].append(b)

# for x in xg:
# 	xg[x].sort(key = lambda x: x[0])

# for y in yg:
# 	yg[y].sort(key = lambda x: x[1])
# print(xg, yg)

# res = 0
# for b in buildings:
# 	x, y = b
# 	if b in xg[x][1:-1] and b in yg[y][1:-1]:
# 		res += 1

# left, right = 1, len(nums)
# while left + 1 < right:
# 	mid = (left + right) // 2
# 	if check(mid):
# 		right = mid
# 	else:
# 		left = mid
# 	return right

# from collections import defaultdict

# nums = [10000]
# start, end = -1, -1
# left = 0
# hash_map = defaultdict(int)
# for right, num in enumerate(nums):
# 	hash_map[num] += 1
# 	while len(hash_map) != right - left + 1:
# 		hash_map[nums[left]] -= 1
# 		if hash_map[nums[left]] == 0:
# 			del hash_map[nums[left]]
# 		left += 1
# 	if end - start < right - left:
# 		start, end = left, right
# print(start, end)
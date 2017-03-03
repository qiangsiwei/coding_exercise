# -*- coding: utf-8 -*- 

# 501 Find Mode in Binary Search Tree
# Given a binary search tree (BST) with duplicates, find all the mode(s) (the most frequently occurred element) in the given BST.
# Assume a BST is defined as follows:
# The left subtree of a node contains only nodes with keys less than or equal to the node's key.
# The right subtree of a node contains only nodes with keys greater than or equal to the node's key.
# Both the left and right subtrees must also be binary search trees.
class Solution:
	def findMode(self, root):
		count = collections.Counter()
		def dfs(node):
			if node:
				count[node.val] += 1
				dfs(node.left)
				dfs(node.right)
		dfs(root)
		max_ct = max(count.itervalues())
		return [k for k, v in count.iteritems() if v == max_ct]

# 502 IPO
# Suppose LeetCode will start its IPO soon. In order to sell a good price of its shares to Venture Capital, LeetCode would like to work on some projects to increase its capital before the IPO. 
# Since it has limited resources, it can only finish at most k distinct projects before the IPO. Help LeetCode design the best way to maximize its total capital after finishing at most k distinct projects.
# You are given several projects. For each project i, it has a pure profit Pi and a minimum capital of Ci is needed to start the corresponding project. Initially, you have W capital. When you finish a project, you will obtain its pure profit and the profit will be added to your total capital.
# To sum up, pick a list of at most k distinct projects from given projects to maximize your final capital, and output your final maximized capital.
# Input: k=2, W=0, Profits=[1,2,3], Capital=[0,1,1].
# Output: 4
class Solution:
	def findMaximizedCapital(self, k, W, Profits, Capital):
		current = []
		future = sorted(zip(Capital, Profits))[::-1]
		for _ in range(k):
			while future and future[-1][0] <= W:
				heapq.heappush(current, -future.pop()[1])
			if current:
				W -= heapq.heappop(current)
		return W

# 503 Next Greater Element II
# Given a circular array (the next element of the last element is the first element of the array), print the Next Greater Number for every element. 
# The Next Greater Number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. If it doesn't exist, output -1 for this number.
class Solution:
	def nextGreaterElements(self, nums):
		st, res = [], [-1]*len(nums)
		for idx, i in enumerate(nums*2):
			while st and (nums[st[-1]] < i):
				res[st.pop()] = i
			if idx < len(nums):
				st.append(idx)
		return res

# 504 Base 7
# Given an integer, return its base 7 string representation.
class Solution:
	def convertTo7(self, num):
		if num == 0: return '0'
		n, res = abs(num), ''
		while n:
		  res = str(n%7)+res; n /= 7
		return res if num >= 0 else '-'+res

# 505 The Maze II 
# There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by rolling up (u), down (d), left (l) or right (r), but it won't stop rolling until hitting a wall. 
# When the ball stops, it could choose the next direction. There is also a hole in this maze. The ball will drop into the hole if it rolls on to the hole.
# Given the ball position, the hole position and the maze, your job is to find out how the ball could drop into the hole by moving shortest distance in the maze.
class Solution:
	def shortestDistance(self, maze, start, destination):
		dest = tuple(destination); m = len(maze); n = len(maze[0]); res = None 
		def go(start, direction):
			i, j = start; ii, jj = direction; l = 0
			while 0<=i+ii<m and 0<=j+jj<n and maze[i+ii][j+jj]!=1:
				i += ii; j += jj; l += 1
			return l, (i,j)
		visited = {}; q = []
		heapq.heappush(q, (0, tuple(start)))
		while q:
			length, cur = heapq.heappop(q)
			if cur in visited and visited[cur] <= length:
				continue # if cur is visited and with a shorter length, skip it.
			visited[cur] = length
			if cur == dest:
				res = min(res, length) if res else length
			for direction in [(-1, 0), (1, 0), (0,-1), (0,1)]:
				l, np = go(cur, direction)
				heapq.heappush(q, (length+l, np))
		return res if res else -1

# 506 Relative Ranks
# Given scores of N athletes, find their relative ranks and the people with the top three highest scores, who will be awarded medals: "Gold Medal", "Silver Medal" and "Bronze Medal".
class Solution:
	def findRelativeRanks(self, nums):
		sort = sorted(nums)[::-1]
		rank = ["Gold Medal", "Silver Medal", "Bronze Medal"] + map(str,range(4,len(nums)+1))
		return map(dict(zip(sort, rank)).get, nums)

# 508 Most Frequent Subtree Sum
# Given the root of a tree, you are asked to find the most frequent subtree sum. The subtree sum of a node is defined as the sum of all the node values formed by the subtree rooted at that node (including the node itself). 
# So what is the most frequent subtree sum value? If there is a tie, return all the values with the highest frequency in any ord
class Solution:
	def findFrequentTreeSum(self, root):
		self.info = collections.defaultdict(int)
		self.calFrequentInfo(root)
		cnt = self.info[max(self.info, key=self.info.get)] if root else 0
		return [key for key, val in self.info.items() if val == cnt]

	def calFrequentInfo(self, root):
		if not root: return 0
		value = root.val + self.calFrequentInfo(root.left) + self.calFrequentInfo(root.right)
		self.info[value] += 1
		return value

# 513 Find Bottom Left Tree Value
# Given a binary tree, find the leftmost value in the last row of the tree.
class Solution:
	def findLeftMostNode(self, root):
		queue = [root]
		for node in queue:
			queue += filter(None, (node.right, node.left))
		return node.val

# 515 Find Largest Value in Each Tree Row
# You need to find the largest value in each row of a binary tree.
class Solution:
	def findValueMostElement(self, root):
		maxes = []; row = [root]
		while any(row):
			maxes.append(max(node.val for node in row))
			row = [kid for node in row for kid in (node.left, node.right) if kid]
		return maxes

# 516 Longest Palindromic Subsequence
# Given a string s, find the longest palindromic subsequence's length in s. You may assume that the maximum length of s is 1000.
class Solution:
	def helper(self, i, j, s, cache):
		if i > j: return 0
		if i == j: return 1
		if i in cache and j in cache[i]:
			return cache[i][j]
		elif s[i] == s[j]:
			cache[i][j] = self.helper(i+1, j-1, s, cache)+2
			return cache[i][j]
		else:
			cache[i][j] = max(self.helper(i, j-1, s, cache), self.helper(i+1, j, s, cache))
			return cache[i][j]	
	def longestPalindromeSubseq(self, s):
		cache = defaultdict(dict)
		return self.helper(0, len(s)-1, s, cache)

# 517 Super Washing Machines
# You have n super washing machines on a line. Initially, each washing machine has some dresses or is empty.
# For each move, you could choose any m (1 ≤ m ≤ n) washing machines, and pass one dress of each washing machine to one of its adjacent washing machines at the same time .
# Given an integer array representing the number of dresses in each washing machine from left to right on the line, you should find the minimum number of moves to make all the washing machines have the same number of dresses.
class Solution:
	def findMinMoves(self, machines):
		if sum(machines) % len(machines) == 0:
			target = sum(machines) / len(machines)
		else:
			return -1
		toLeft = 0; res = 0
		for i in range(len(machines)):
			toRight = machines[i]-target-toLeft
			res = max(res, toLeft, toRight, toLeft+toRight)
			toLeft = -toRight
		return res

# 520 Detect Capital
# Given a word, you need to judge whether the usage of capitals in it is right or not.
# We define the usage of capitals in a word to be right when one of the following cases holds:
# All letters in this word are capitals, like "USA".
# All letters in this word are not capitals, like "leetcode".
# Only the first letter in this word is capital if it has more than one letter, like "Google".
# Otherwise, we define that this word doesn't use capitals in a right way.
class Solution:
	def detectCapitalUse(self, word):
		total = sum(map(lambda x:x.isupper(), word))
		return total == 0 or len(word) == total or (total == 1 and word[0].isupper())

# 525 Contiguous Array
# Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1
class Solution:
	def findMaxLength(self, nums):
		count = 0; max_length = 0; table = {0: 0}
		for index, num in enumerate(nums, 1):
			if num == 0:
				count -= 1
			else:
				count += 1
			if count in table:
				max_length = max(max_length, index-table[count])
			else:
				table[count] = index
		return max_length

# 526 Beautiful Arrangement
# Suppose you have N integers from 1 to N. We define a beautiful arrangement as an array that is constructed by these N numbers successfully if one of the following is true for the ith position (1 ≤ i ≤ N) in this array:
# The number at the ith position is divisible by i.
# i is divisible by the number at the ith position.
# Now given N, how many beautiful arrangements can you construct?
class Solution:
	def countArrangement(self, N):
		def count(i, X):
			if i == 1: return 1
			return sum(count(i-1, X-{x}) for x in X
					   if x % i == 0 or i % x == 0)
		return count(N, set(range(1, N+1)))

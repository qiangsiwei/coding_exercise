# -*- coding: utf-8 -*- 

# 301 Remove Invalid Parentheses
# Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.
class Solution:
	def removeInvalidParentheses(self, s):
		def isvalid(s):
			ctr = 0
			for c in s:
				if c == '(':
					ctr += 1
				elif c == ')':
					ctr -= 1
					if ctr < 0:
						return False
			return ctr == 0
		level = {s}
		while True:
			valid = filter(isvalid, level)
			if valid:
				return valid
			level = {s[:i]+s[i+1:] for s in level for i in range(len(s))}

# 302 Smallest Rectangle Enclosing Black Pixels
class Solution:
	def minArea(self, image, x, y): # Binary Search
		top = self.searchRows(image, 0, x, True)
		bottom = self.searchRows(image, x+1, len(image), False)
		left = self.searchColumns(image, 0, y, top, bottom, True)
		right = self.searchColumns(image, y+1, len(image[0]), top, bottom, False)
		return (right-left)*(bottom-top)
	def searchRows(self, image, i, j, opt):
		while i != j:
			m = (i+j)/2
			if any(p == '1' for p in image[m]) == opt: j = m
			else: i = m+1
		return i
	def searchColumns(self, image, i, j, top, bottom, opt):
		while i != j:
			m = (i+j)/2
			if any(image[k][m] == '1' for k in xrange(top, bottom)) == opt: j = m
			else: i = m+1
		return i

# 303 Range Sum Query - Immutable
# Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
class NumArray:
	def __init__(self, nums):
		size = len(nums)
		self.sums = [0]*(size+1)
		for x in range(size):
			self.sums[x+1] += self.sums[x]+nums[x]
	def sumRange(self, i, j):
		return self.sums[j+1]-self.sums[i]

# 304 Range Sum Query 2D - Immutable
# Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
class NumMatrix:
	def __init__(self, matrix):
		m = len(matrix)
		n = len(matrix[0]) if m else 0
		self.sums = [[0]*(n+1) for x in range(m+1)]
		for x in range(1, m+1):
			rowSum = 0
			for y in range(1, n+1):
				self.sums[x][y] += rowSum+matrix[x-1][y-1]
				if x > 1:
					self.sums[x][y] += self.sums[x-1][y]
				rowSum += matrix[x-1][y-1]
	def sumRegion(self, row1, col1, row2, col2):
		return self.sums[row2+1][col2+1] + self.sums[row1][col1] \
			 - self.sums[row1][col2+1] - self.sums[row2+1][col1]

# 305 Number of Islands II
# Given a n,m which means the row and column of the 2D matrix and an array of pair A(size k). Originally, the 2D matrix is all 0 which means there is only sea in the matrix. 
# The list pair has k operator and each operator has two integer A[i].x, A[i].y means that you can change the grid matrix[A[i].x][A[i].y] from sea to island. 
# Return how many island are there in the matrix after each operator.
class Solution:
	def numIslands2(self, m, n, positions):
		parent, rank = {}, {}
		def find(x):
			if parent[x] != x:
				parent[x] = find(parent[x])
			return parent[x]
		def union(x, y):
			x, y = find(x), find(y)
			if x == y:
				return 0
			if rank[x] < rank[y]:
				x, y = y, x
			parent[y] = x
			rank[x] += rank[x] == rank[y]
			return 1
		counts, count = [], 0
		for i, j in positions:
			x = parent[x] = i, j
			rank[x] = 0
			count += 1
			for y in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
				if y in parent:
					count -= union(x, y)
			counts.append(count)
		return counts

# 306 Additive Number
# Additive number is a string whose digits can form additive sequence.
class Solution:
	def isAdditiveNumber(self, num):
		length = len(num)
		for i in range(1, length/2+1):
			for j in range(1, (length-i)/2+1):
				first, second, others = num[:i], num[i:i+j], num[i+j:]
				if self.isValid(first, second, others):
					return True
		return False
	def isValid(self, first, second, others):
		if ((len(first) > 1 and first[0] == "0") or
			(len(second) > 1 and second[0] == "0")):
			return False
		sum_str = str(int(first)+int(second))
		if sum_str == others:
			return True
		elif others.startswith(sum_str):
			return self.isValid(second, sum_str, others[len(sum_str):])
		else:
			return False

# 307 Range Sum Query - Mutable
# Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
# The update(i, val) function modifies nums by updating the element at index i to val.
class NumArray: # 树状数组
	def __init__(self, nums):
		self.n = len(nums)
		self.a, self.c = nums, [0]*(self.n+1)
		for i in range(self.n):
			k = i+1
			while k <= self.n:
				self.c[k] += nums[i]
				k += (k&-k)
	def update(self, i, val):
		diff, self.a[i] = val-self.a[i], val
		i += 1
		while i <= self.n:
			self.c[i] += diff
			i += (i&-i)
	def sumRange(self, i, j):
		res, j = 0, j+1
		while j:
			res += self.c[j]
			j -= (j&-j)
		while i:
			res -= self.c[i]
			i -= (i&-i)
		return res

# 308 Range Sum Query 2D - Mutable
class NumMatrix:
	def __init__(self, matrix):
		if not matrix:
			return
		self.m, self.n = len(matrix), len(matrix[0])
		self.matrix, self.bit = [[0]*(self.n) for _ in range(self.m)], [[0]*(self.n+1) for _ in range(self.m+1)]
		for i in range(self.m):
			for j in range(self.n):
				self.update(i, j, matrix[i][j])
	def update(self, row, col, val):
		diff, self.matrix[row][col], i = val-self.matrix[row][col], val, row+1
		while i <= self.m:
			j = col+1
			while j <= self.n:
				self.bit[i][j] += diff
				j += (j&-j)
			i += (i&-i)
	def sumRegion(self, row1, col1, row2, col2):
		return self.sumCorner(row2, col2) + self.sumCorner(row1-1, col1-1) \
			 - self.sumCorner(row1-1, col2) - self.sumCorner(row2, col1-1)
	def sumCorner(self, row, col):
		res, i = 0, row+1
		while i:
			j = col+1
			while j:
				res += self.bit[i][j]
				j -= (j&-j)
			i -= (i&-i)
		return res

# 309 Best Time to Buy and Sell Stock with Cooldown
# You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:
# You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
# After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
# notHold -- buy --> hold
# hold -- sell --> cooldown
# hold -- do nothing --> hold
# notHold -- do nothing --> notHold
# cooldown -- do nothing --> notHold
class Solution:
	def maxProfit(self, prices):
		notHold, notHold_cooldown, hold = 0, float('-inf'), float('-inf')
		for p in prices:
			hold, notHold, notHold_cooldown = max(hold, notHold-p), max(notHold, notHold_cooldown), hold+p
		return max(notHold, notHold_cooldown)

# 310 Minimum Height Trees
# For a undirected graph with tree characteristics, we can choose any node as the root. 
# The result graph is then a rooted tree. Among all possible rooted trees, those with minimum height are called minimum height trees (MHTs). 
class Solution:
	def findMinHeightTrees(self, n, edges):
		children = collections.defaultdict(set)
		for s, t in edges:
			children[s].add(t)
			children[t].add(s)
		vertices = set(children.keys())
		while len(vertices) > 2:
			leaves = [x for x in children if len(children[x]) == 1]
			for x in leaves:
				for y in children[x]:
					children[y].remove(x)
				del children[x]
				vertices.remove(x)
		return list(vertices) if n != 1 else [0]

# 311 Sparse Matrix Multiplication
# Given two sparse matrices A and B, return the result of AB.
class Solution:
	def multiply(self, A, B):
		result = [[0]*len(B[0]) for _ in xrange(len(A))]
		Bi = [[i for i,v in enumerate(l) if v] if any(l) else [] for l in B]
		for i in xrange(len(A)):
			if not any(A[i]):
				continue
			for j in xrange(len(A[0])):
				if A[i][j]:
					for k in Bi[j]:
						result[i][k]+=A[i][j]*B[j][k]
		return result

# 312 Burst Balloons
# Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums. 
# You are asked to burst all the balloons. If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins. 
# Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.
# Find the maximum coins you can collect by bursting the balloons wisely.
class Solution:
	def maxCoins(self, nums): # bottom up with rescursive
		nums = [1]+nums+[1]; n = len(nums)
		dp = [[0]*n for _ in xrange(n)]
		def helper(left, right):
			if right-left <=1: # if there are 1 or 2 elements
				return 0
			if dp[left][right]:
				return dp[left][right]
			res = []
			for last in range(left+1, right): # choose which element to finish lastly and caculate sub array
				res.append(nums[left]*nums[last]*nums[right]+helper(left, last)+helper(last,right))
			dp[left][right] = max(res)
			return dp[left][right]
		return helper(0,len(nums)-1)

# 313 Super Ugly Number
# Write a program to find the nth super ugly number.
# Super ugly numbers are positive numbers whose all prime factors are in the given prime list primes of size k. 
# For example, [1, 2, 4, 7, 8, 13, 14, 16, 19, 26, 28, 32] is the sequence of the first 12 super ugly numbers given primes = [2, 7, 13, 19] of size 4.
class Solution:
	def nthSuperUglyNumber(self, n, primes):
		uglies = [1]
		def gen(prime):
			for ugly in uglies:
				yield ugly * prime
		merged = heapq.merge(*map(gen, primes))
		while len(uglies) < n:
			ugly = next(merged)
			if ugly != uglies[-1]:
				uglies.append(ugly)
		return uglies[-1]

# 314 Binary Tree Vertical Order Traversal
# Given a binary tree, return the vertical order traversal of its nodes' values. (ie, from top to bottom, column by column).
# If two nodes are in the same row and column, the order should be from left to right.
class Solution:
	def verticalOrder(self, root):
		if not root: return []
		d = {}; queue = [(root, 0)]
		# index denotes which column the node is in
		while queue:
			curNode, index = queue.pop(0)
			d[index] = d.get(index, [])+[curNode.val]
			if curNode.left: queue.append((curNode.left, index-1))
			if curNode.right: queue.append((curNode.right, index+1))
		res = []; minIndex = min(d.keys()); maxIndex = max(d.keys())
		for i in range(minIndex, maxIndex+1):
			res.append(d[i])
		return res

# 315 Count of Smaller Numbers After Self
# You are given an integer array nums and you have to return a new counts array. 
# The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].
class Solution:
	def countSmaller(self, nums):
		def sort(enum):
			half = len(enum)/2
			if half:
				left, right = sort(enum[:half]), sort(enum[half:])
				for i in range(len(enum))[::-1]:
					if not right or left and left[-1][1] > right[-1][1]:
						smaller[left[-1][0]] += len(right)
						enum[i] = left.pop()
					else:
						enum[i] = right.pop()
			return enum
		smaller = [0] * len(nums)
		sort(list(enumerate(nums)))
		return smaller

# 316 Remove Duplicate Letters
# Given a string which contains only lowercase letters, remove duplicate letters so that every letter appear once and only once. 
# You must make sure your result is the smallest in lexicographical order among all possible results.
class Solution:
	def removeDuplicateLetters(self, s):
		for c in sorted(set(s)):
			suffix = s[s.index(c):]
			if set(suffix) == set(s):
				return c+self.removeDuplicateLetters(suffix.replace(c,''))
		return ''

# 317 Shortest Distance from All Buildings
# You want to build a house on an empty land which reaches all buildings in the shortest amount of distance. You can only move up, down, left and right. You are given a 2D grid of values 0, 1 or 2, where:
# Each 0 marks an empty land which you can pass by freely.
# Each 1 marks a building which you cannot pass through.
# Each 2 marks an obstacle which you cannot pass through.
# For example, given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2):
# 1 - 0 - 2 - 0 - 1
# |   |   |   |   |
# 0 - 0 - 0 - 0 - 0
# |   |   |   |   |
# 0 - 0 - 1 - 0 - 0
# The point (1,2) is an ideal empty land to build a house, as the total travel distance of 3+3+1=7 is minimal. So return 7.
# Note: There will be at least one building. If it is not possible to build such house according to the above rules, return -1.
class Solution:
	def traverse(self, startPoint, m, n, grid, distance):
		queue, level, nth = collections.deque([startPoint]), 1, self.nth
		while queue:
			for _ in xrange(len(queue)):
				i, j = queue.popleft()
				for x, y in ((0, -1), (-1, 0), (0, 1), (1, 0)):
					ii, jj = i+x, j+y
					if 0 <= ii < m and 0 <= jj < n and grid[ii][jj] == nth+1:
						queue.append((ii, jj))
						distance[ii][jj] += level
						grid[ii][jj] = nth
			level += 1
		self.nth -= 1
	def shortestDistance(self, grid):
		m, n, self.nth = len(grid), len(grid[0]), -1
		distance = [[0]*n for _ in xrange(m)]
		buildingNumber = len([
			self.traverse((i, j), m, n, grid, distance)
			for i, row in enumerate(grid)
			for j, num in enumerate(row) if num == 1
		])
		return min([
			distance[i][j]
			for i, row in enumerate(grid)
			for j, num in enumerate(row)
			if num == -buildingNumber
		] or [-1])

# 318 Maximum Product of Word Lengths
# Given a string array words, find the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters. 
# You may assume that each word will contain only lower case letters. If no such two words exist, return 0.
class Solution:
	def maxProduct(self, words):
		nums = []
		size = len(words)
		for w in words:
			nums += sum(1<<(ord(x)-ord('a')) for x in set(w)),
		ans = 0
		for x in range(size):
			for y in range(size):
				if not (nums[x] & nums[y]):
					ans = max(len(words[x]) * len(words[y]), ans)
		return ans

# 319 Bulb Switcher
# There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second bulb. On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). 
# For the ith round, you toggle every i bulb. For the nth round, you only toggle the last bulb. Find how many bulbs are on after n rounds.
class Solution:
	def bulbSwitch(self, n):
		return int(math.sqrt(n))

# 320 Generalized Abbreviation
# Write a function to generate the generalized abbreviations of a word.
# Example:
# Given word = "word", return the following list (order does not matter):
# ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]
class Solution:
	def generateAbbreviations(self, word):
		return [word] + [word[:first] + str(last-first+1) + word[last+1:last+2] + rest
							for last in range(len(word))
							for first in range(last+1)
							for rest in self.generateAbbreviations(word[last+2:])]

# 321 Create Maximum Number
# Given two arrays of length m and n with digits 0-9 representing two numbers. Create the maximum number of length k <= m + n from digits of the two. 
# The relative order of the digits from the same array must be preserved. Return an array of the k digits. You should try to optimize your time and space complexity.
class Solution:
	def maxNumber(self, nums1, nums2, k):
		def getMax(nums, t):
			ans = []
			size = len(nums)
			for x in range(size):
				while ans and len(ans)+(size-x)>t and ans[-1]<nums[x]:
					ans.pop()
				if len(ans) < t:
					ans += nums[x],
			return ans
		def merge(nums1, nums2):
			ans = []
			while nums1 or nums2:
				if nums1 > nums2:
					ans += nums1[0],
					nums1 = nums1[1:]
				else:
					ans += nums2[0],
					nums2 = nums2[1:]
			return ans
		len1, len2 = len(nums1), len(nums2)
		res = []
		for x in range(max(0,k-len2), min(k,len1)+1):
			tmp = merge(getMax(nums1,x), getMax(nums2,k-x))
			res = max(tmp, res)
		return res

# 322 Coin Change
# You are given coins of different denominations and a total amount of money amount. 
# Write a function to compute the fewest number of coins that you need to make up that amount. 
# If that amount of money cannot be made up by any combination of the coins, return -1.
class Solution:
	def coinChange(self, coins, amount):
		dp = [0] + [-1]*amount
		for x in range(amount):
			if dp[x] < 0:
				continue
			for c in coins:
				if x+c > amount:
					continue
				if dp[x+c] < 0 or dp[x+c] > dp[x]+1:
					dp[x+c] = dp[x]+1
		return dp[amount]

# 323 Number of Connected Components in an Undirected Graph
# Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), 
# write a function to find the number of connected components in an undirected graph.
class Solution:
	import collections
	def depthFirstSearch(self, adj_list, visited, vertex):
		visited[vertex] = True
		for neighbor in adj_list[vertex]:
			if not visited[neighbor]:
				self.depthFirstSearch(adj_list, visited, neighbor)
	def countComponents(self, n, edges):
		adj_list = collections.defaultdict(list)
		for edge in edges:
			adj_list[edge[0]].append(edge[1])
			adj_list[edge[1]].append(edge[0])
		visited = [False for i in xrange(n)]
		connected_components = 0
		for vertex in xrange(n):
			if not visited[vertex]:
				self.depthFirstSearch(adj_list, visited, vertex)
				connected_components += 1
		return connected_components

# 324 Wiggle Sort II
# Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]....
class Solution:
	def wiggleSort(self, nums):
		size = len(nums)
		snums = sorted(nums)
		for x in range(1, size, 2) + range(0, size, 2):
			nums[x] = snums.pop()

# 325 Maximum Size Subarray Sum Equals k
# Given an array nums and a target value k, find the maximum length of a subarray that sums to k. If there isn't one, return 0 instead.
class Solution:
	def maxSubArrayLen(self, nums, k):
		ans, acc, table = 0, 0, {0:-1}
		for i in xrange(len(nums)):
			acc += nums[i]
			if acc not in table:
				table[acc] = i 
			if acc-k in table:
				ans = max(ans, i-table[acc-k])
		return ans

# 326 Power of Three
# Given an integer, write a function to determine if it is a power of three.
class Solution:
	def isPowerOfThree(self, n):
		return n > 0 and 1162261467 % n == 0

# 327 Count of Range Sum
# Given an integer array nums, return the number of range sums that lie in [lower, upper] inclusive.
# Range sum S(i, j) is defined as the sum of the elements in nums between indices i and j (i ≤ j), inclusive.
class Solution:
	def countRangeSum(self, nums, lower, upper):
		first = [0]
		for num in nums:
			first.append(first[-1]+num)
		def sort(lo, hi):
			mid = (lo+hi)/2
			if mid == lo:
				return 0
			count = sort(lo,mid) + sort(mid,hi)
			i = j = mid
			for left in first[lo:mid]:
				while i < hi and first[i]-left <  lower: i += 1
				while j < hi and first[j]-left <= upper: j += 1
				count += j-i
			first[lo:hi] = sorted(first[lo:hi])
			return count
		return sort(0, len(first))

# 328 Odd Even Linked List
# Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.
class Solution:
	def oddEvenList(self, head):
		if head is None: return head
		odd = oddHead = head
		even = evenHead = head.next
		while even and even.next:
			odd.next = even.next
			odd = odd.next
			even.next = odd.next
			even = even.next
		odd.next = evenHead
		return oddHead

# 329 Longest Increasing Path in a Matrix
# Given an integer matrix, find the length of the longest increasing path.
# From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).
class Solution:
	def longestIncreasingPath(self, matrix):
		matrix = {i+j*1j: val
				  for i, row in enumerate(matrix)
				  for j, val in enumerate(row)}
		length = {}
		for z in sorted(matrix, key=matrix.get):
			length[z] = 1 + max([length[Z] for Z in z+1, z-1, z+1j, z-1j
									if Z in matrix and matrix[z] > matrix[Z]]
								or [0])
		return max(length.values() or [0])

# 330 Patching Array
# Given a sorted positive integer array nums and an integer n, add/patch elements to the array such that any number in range [1, n] 
# inclusive can be formed by the sum of some elements in the array. Return the minimum number of patches required.
class Solution:
	def minPatches(self, nums, n):
		miss, i, added = 1, 0, 0
		while miss <= n:
			if i < len(nums) and nums[i] <= miss:
				miss += nums[i]
				i += 1
			else:
				miss += miss
				added += 1
		return added

# 331 Verify Preorder Serialization of a Binary Tree
# Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of a binary tree. Find an algorithm without reconstructing the tree.
class Solution:
	def isValidSerialization(self, preorder):
		stack = collections.deque()
		for item in preorder.split(','):
			stack.append(item)
			while len(stack) >= 3 and \
				  stack[-1] == stack[-2] == '#' and \
				  stack[-3] != '#':
				stack.pop(), stack.pop(), stack.pop()
				stack.append('#')
		return len(stack) == 1 and stack[0] == '#'

# 332 Reconstruct Itinerary
# Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. 
# All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.
class Solution:
	def findItinerary(self, tickets):
		targets = collections.defaultdict(list)
		for a, b in sorted(tickets)[::-1]:
			targets[a] += b,
		route = []
		def visit(airport):
			while targets[airport]:
				visit(targets[airport].pop())
			route.append(airport)
		visit('JFK')
		return route[::-1]

# 333 Largest BST Subtree
# Given a binary tree, find the largest subtree which is a Binary Search Tree (BST), where largest means subtree with largest number of nodes in it.
class SubTree:
	def __init__(self, largest, n, min, max):
		self.largest = largest	# largest BST
		self.n = n				# number of nodes in this ST
		self.min = min			# min val in this ST
		self.max = max			# max val in this ST
class Solution:
	def largestBSTSubtree(self, root):
		res = self.dfs(root)
		return res.largest
	def dfs(self, root):
		if not root:
			return SubTree(0, 0, float('inf'), float('-inf'))
		left = self.dfs(root.left)
		right = self.dfs(root.right)
		if root.val > left.max and root.val < right.min:  # valid BST
			n = left.n + right.n + 1
		else:
			n = float('-inf')
		largest = max(left.largest, right.largest, n)
		return SubTree(largest, n, min(left.min, root.val), max(right.max, root.val))

# 334 Increasing Triplet Subsequence
# Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.
class Solution:
	def increasingTriplet(self, nums):
		a = b = None
		for n in nums:
			if a is None or a >= n:
				a = n
			elif b is None or b >= n:
				b = n
			else:
				return True
		return False

# 335 Self Crossing
# You are given an array x of n positive numbers. 
# You start at point (0,0) and moves x[0] metres to the north, then x[1] metres to the west, x[2] metres to the south, x[3] metres to the east and so on. 
# In other words, after each move your direction changes counter-clockwise.
#		   b							  b
#   +----------------+			 +----------------+
#   |				|			 |				|
#   |				|			 |				|
# c |				| a		 c |				| a
#   |				|			 |				|   f
#   +----------->	|			 |				| <----+
#		 d		  |			 |				|	  | e
#					|			 |					   |
#								  +-----------------------+
#											  d
class Solution:
	def isSelfCrossing(self, x):
		b = c = d = e = 0
		for a in x:
			if d >= b > 0 and (a >= c or a >= c-e >= 0 and f >= d-b):
				return True
			b, c, d, e, f = a, b, c, d, e
		return False

# 336 Palindrome Pairs
# Given a list of unique words. Find all pairs of distinct indices (i, j) in the given list, 
# so that the concatenation of the two words, i.e. words[i] + words[j] is a palindrome.
class Solution:
	def palindromePairs(self, words):
		wordict, res = {}, []
		for i in range(len(words)):
			wordict[words[i]] = i
		for i in range(len(words)):
			for j in range(len(words[i])+1):
				tmp1, tmp2 = words[i][:j], words[i][j:]
				if tmp1[::-1] in wordict and wordict[tmp1[::-1]]!=i and tmp2 == tmp2[::-1]:
					res.append([i,wordict[tmp1[::-1]]])
				if tmp2[::-1] in wordict and wordict[tmp2[::-1]]!=i and tmp1 == tmp1[::-1] and j!=0:
					res.append([wordict[tmp2[::-1]],i])
		return res

# 337 House Robber III
# The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root." Besides the root, each house has one and only one parent house. 
# After a tour, the smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses were broken into on the same night.
# Determine the maximum amount of money the thief can rob tonight without alerting the police.
class Solution:
	def rob(self, root):
		valMap = {}
		def solve(root, path):
			if root is None: return 0
			if path not in valMap:
				left, right = root.left, root.right
				ll = lr = rl = rr = None
				if left:  ll, lr = left.left, left.right
				if right: rl, rr = right.left, right.right
				passup = solve(left, path+'l') + solve(right, path+'r')
				grabit = root.val + solve(ll, path+'ll') + solve(lr, path+'lr') \
								  + solve(rl, path+'rl') + solve(rr, path+'rr')
				valMap[path] = max(passup, grabit)
			return valMap[path]
		return solve(root, '')

# 338 Counting Bits
# Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.
class Solution:
	def countBits(self, num):
		ans = [0]
		for x in range(1, num+1):
			ans += ans[x>>1]+(x&1),
		return ans

# 339 Nested List Weight Sum
# Given a nested list of integers, return the sum of all integers in the list weighted by their depth.
# Each element is either an integer, or a list -- whose elements may also be integers or other lists.
class Solution:
	def depthSum(self, nestedList):
		def DFS(nestedList, depth):
			tmp_sum = 0
			for member in nestedList:
				if member.isInteger():
					tmp_sum += member.getInteger()*depth
				else:
					tmp_sum += DFS(member.getList(), depth+1)
			return tmp_sum
		return DFS(nestedList,1)

# 340 Longest Substring with At Most K Distinct Characters 
# Given a string, find the length of the longest substring T that contains at most k distinct characters.
# For example, Given s = "eceba" and k = 2,
# T is "ece" which its length is 3.
class Solution:
	def lengthOfLongestSubstringKDistinct(s, k):
		tail, seen, lenmax = 0, {s[0]:1}, 0
		for head in xrange(1,len(s)):
			if s[head] in seen or len(seen) < k:
				seen[s[head]] = seen.get(s[head],0)+1
				lenmax = max(lenmax,head-tail+1)
			else:
				seen[s[tail]] -= 1
				if seen[s[tail]] == 0:
					del seen[s[tail]]
				tail += 1
		print lenmax

# 341 Flatten Nested List Iterator
# Given a nested list of integers, implement an iterator to flatten it.
# Each element is either an integer, or a list -- whose elements may also be integers or other lists.
class NestedIterator:
	def __init__(self, nestedList):
		self.stack = [[nestedList, 0]]
	def next(self):
		self.hasNext()
		nestedList, i = self.stack[-1]
		self.stack[-1][1] += 1
		return nestedList[i].getInteger()
	def hasNext(self):
		s = self.stack
		while s:
			nestedList, i = s[-1]
			if i == len(nestedList):
				s.pop()
			else:
				x = nestedList[i]
				if x.isInteger():
					return True
				s[-1][1] += 1
				s.append([x.getList(), 0])
		return False

# 342 Power of Four
# Given an integer (signed 32 bits), write a function to check whether it is a power of 4.
class Solution:
	def isPowerOfFour(self, num):
		return num > 0 and bin(num).count('0') % 2 == 1 and bin(num).count('1') == 1

# 343 Integer Break
# Given a positive integer n, break it into the sum of at least two positive integers 
# and maximize the product of those integers. Return the maximum product you can get.
class Solution:
	def integerBreak(self, n):
		prod = []
		for i in range(2,n+1):
			x = round(n/float(i))
			y = (x**(i-1))*(n-x*(i-1))
			prod.append(int(y))
		return max(prod)

# 344 Reverse String
# Write a function that takes a string as input and returns the string reversed.
class Solution:
	def reverseString(self, s):
		return s[::-1]

# 345 Reverse Vowels of a String
# Write a function that takes a string as input and reverse only the vowels of a string.
class Solution:
	def reverseVowels(self, s):
		vow = {"a","e","i","o","u","A","E","I","O","U"}
		s, index, word = list(s), [], []
		for i in range(len(s)):
			if s[i] in vow:
				index.append(i)
				word.append(s[i])
		for i in index:
			s[i] = word.pop()
		return "".join(s)

# 346 Moving Average from Data Stream
# Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

class MovingAverage:
	import collections
	def __init__(self, size):
		self.queue = collections.deque(maxlen=size)
	def next(self, val):
		queue = self.queue
		queue.append(val)
		return float(sum(queue))/len(queue)

# 347 Top K Frequent Elements
# Given a non-empty array of integers, return the k most frequent elements.
class Solution:
	import heapq
	from collections import Counter
	def topKFrequent(self, nums, k):
		c = Counter(nums)
		return heapq.nlargest(k, c, key=lambda x:c[x])

# 348 Design Tic-Tac-Toe
# Design a Tic-tac-toe game that is played between two players on a n x n grid.
# You may assume the following rules:
# A move is guaranteed to be valid and is placed on an empty block.
# Once a winning condition is reached, no more moves is allowed.
# A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.
class TicTacToe:
	def __init__(self, n):
		count = collections.Counter()
		def move(row, col, player):
			for i, x in enumerate((row, col, row+col, row-col)):
				count[i, x, player] += 1
				if count[i, x, player] == n:
					return player
			return 0
		self.move = move

# 349 Intersection of Two Arrays
# Given two arrays, write a function to compute their intersection.
# Each element in the result must be unique.
class Solution:
	def intersection(self, nums1, nums2):
		return list(set(nums1).intersection(set(nums2)))

# 350 Intersection of Two Arrays II
# Each element in the result should appear as many times as it shows in both arrays.
class Solution:
	from collections import Counter
	def intersect(self, nums1, nums2):
		c1, c2 = Counter(nums1), Counter(nums2)
		return sum([[num]*min(c1[num],c2[num]) for num in c1 & c2], [])

# 351 Android Unlock Patterns
# Given an Android 3x3 key lock screen and two integers m and n, where 1 ≤ m ≤ n ≤ 9, count the total number of unlock patterns of the Android lock screen, which consist of minimum of m keys and maximum n keys.
# Rules for a valid pattern: Each pattern must connect at least m keys and at most n keys. All the keys must be distinct.
# If the line connecting two consecutive keys in the pattern passes through any other keys, the other keys must have previously selected in the pattern. No jumps through non selected key is allowed.
class Solution:
	def numberOfPatterns(self, m, n):
		visited, ret = {}, 0
		self.dfsHelper(ret, visited, 0, m, n, -9)
		return ret
	def dfsHelper(self, ret, visited, res, m, n, prev):
		if m <= res <= n:
			ret += 1
		if res == n:
			return
		for i in xrange(1, 10):
			if i not in visited:
				x, y, xp, yp = (i-1)/3, (i-1)%3, (prev-1)/3, (prev-1)%3
				if (5 not in visited and abs(xp-x) == 2 and abs(yp-y) == 2) or \
				   ((yp == y and abs(xp-x) == 2) or (xp == x and abs(yp-y) == 2)) and (prev+i)/2 not in visited:
					continue
				visited[i] = 1
				self.dfsHelper(ret, visited, res+1, m, n, i)
				del visited[i]

# 352 Data Stream as Disjoint Intervals
# Given a data stream input of non-negative integers a1, a2, ..., an, ..., summarize the numbers seen so far as a list of disjoint intervals.
# For example, suppose the integers from the data stream are 1, 3, 7, 2, 6, ..., then the summary will be:
# [1, 1]
# [1, 1], [3, 3]
# [1, 1], [3, 3], [7, 7]
# [1, 3], [7, 7]
# [1, 3], [6, 7]
class SummaryRanges:
	def __init__(self):
		self.have, self.left, self.right = set(), {}, {}
	def addNum(self, val):
		if val in self.have: return
		self.have.add(val)
		if val-1 in self.right and val+1 in self.left:
			l, r = self.right.pop(val-1), self.left.pop(val+1)
			self.left.pop(l.start)
			self.right.pop(r.end)
			interval = Interval(l.start, r.end)
			self.left[l.start], self.right[r.end] = interval, interval
		elif val-1 in self.right:
			l = self.right.pop(val-1)
			self.left.pop(l.start)
			interval = Interval(l.start, val)
			self.left[l.start], self.right[val] = interval, interval
		elif val+1 in self.left:
			r = self.left.pop(val+1)
			self.right.pop(r.end)
			interval = Interval(val, r.end)
			self.left[val], self.right[r.end] = interval, interval
		else:
			interval = Interval(val, val)
			self.left[val], self.right[val] = interval, interval
	def getIntervals(self):
		return [self.left[key] for key in sorted(self.left.keys())]

# 353 Design Snake Game
# Design a Snake game that is played on a device with screen size = width x height. Play the game online if you are not familiar with the game.
# The snake is initially positioned at the top left corner (0,0) with length = 1 unit.
# You are given a list of food's positions in row-column order. When a snake eats the food, its length and the game's score both increase by 1.
# Each food appears one by one on the screen. For example, the second food will not appear until the first food was eaten by the snake.
# When a food does appear on the screen, it is guaranteed that it will not appear on a block occupied by the snake.
class SnakeGame:
	def __init__(self, width, height,food):
		self.m, self.n = height, width
		self.moves = {"U":-1, "D":+1, "L":-1j, "R":1j}
		self.setsnake, self.snake = {0j}, collections.deque([0j])
		self.food = [r+c*1j for r, c in food[::-1]]
		self.curfood = self.food.pop() if self.food else None
	def move(self, direc):
		move = self.snake[-1] + self.moves[direc]
		tail = self.snake.popleft()
		self.setsnake.discard(tail)
		if 0 <= move.real < self.m and 0 <= move.imag < self.n and move not in self.setsnake:
			self.snake.append(move)
			self.setsnake.add(move)
			if self.curfood == move:
				self.curfood = self.food.pop() if self.food else None
				self.snake.appendleft(tail)
				self.setsnake.add(tail)
			return len(self.snake)-1
		return -1

# 354 Russian Doll Envelopes
# You have a number of envelopes with widths and heights given as a pair of integers (w, h). 
# One envelope can fit into another if and only if both the width and height of one envelope is greater than the width and height of the other envelope.
# What is the maximum number of envelopes can you Russian doll? (put one inside other)
class Solution:
	def maxEnvelopes(self, envelopes):
		des_ht = [a[1] for a in sorted(envelopes, key=lambda x:(x[0],-x[1]))]
		dp, l = [0]*len(des_ht), 0
		for x in des_ht:
			i = bisect.bisect_left(dp, x, 0, l)
			dp[i] = x
			if i == l:
				l += 1
		return l

# 355 Design Twitter
# Design a simplified version of Twitter where users can post tweets, follow/unfollow another user and is able to see the 10 most recent tweets in the user's news feed. Your design should support the following methods:
# postTweet(userId, tweetId): Compose a new tweet.
# getNewsFeed(userId): Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
# follow(followerId, followeeId): Follower follows a followee.
# unfollow(followerId, followeeId): Follower unfollows a followee.
class Twitter:
	def __init__(self):
		self.timer = itertools.count(step=-1)
		self.tweets = collections.defaultdict(collections.deque)
		self.followees = collections.defaultdict(set)
	def postTweet(self, userId, tweetId):
		self.tweets[userId].appendleft((next(self.timer), tweetId))
	def getNewsFeed(self, userId):
		tweets = heapq.merge(*(self.tweets[u] for u in self.followees[userId]|{userId}))
		return [t for _, t in itertools.islice(tweets, 10)]
	def follow(self, followerId, followeeId):
		self.followees[followerId].add(followeeId)
	def unfollow(self, followerId, followeeId):
		self.followees[followerId].discard(followeeId)

# 356 Line Reflection
# Given n points on a 2D plane, find if there is such a line parallel to y-axis that reflect the given set of points.
# Example 1:
# Given points = [[1,1],[-1,1]], return true.
# Example 2:
# Given points = [[1,1],[-1,-1]], return false.
class Solution:
	def isReflected(self, points):
		if not points: return True
		X = min(points)[0] + max(points)[0]
		return {(x, y) for x, y in points} == {(X-x, y) for x, y in points}

# 357 Count Numbers with Unique Digits
# Given a non-negative integer n, count all numbers with unique digits, x, where 0 ≤ x < 10n.
class Solution:
	def countNumbersWithUniqueDigits(self, n):
			# n can not be greater than 10 as it must exist two of the digit share same number
			# f(n) = 1-digits unique num combination + 2-digits unique num combination + ...
			# f(n) = 10 + 9 * 9 + 9 * 9 * 8 + ...
			# f(n) = g(0) + g(1) + g(2) + ... + g(n)
			# g(0) = 10
			# g(1) = 9 * 9
			# g(2) = 9 * 9 * 8
			# g(k) = 9 * (10 - 1) * (10 - 2) * ... * (10 - k)
			if n == 0: return 1
			if n == 1: return 10
			n = min(10, n)
			res = 10
			for n in range(1, n):
				g = 9
				for i in range(1, n+1):
					g *= (10 - i)
				res += g
			return res

# 358 Rearrange String k Distance Apart
# Given a non-empty string str and an integer k, rearrange the string such that the same characters are at least distance k from each other.
# All input strings are given in lowercase letters. If it is not possible to rearrange the string, return an empty string "".
class Solution:
	import heapq
	def rearrangeString(self, str, k):
		heap = [(-freq, char) for char, freq in collections.Counter(str).items()]
		heapq.heapify(heap)
		res = []
		while len(res) < len(str):
			if not heap: return ""
			freq, char = heapq.heappop(heap)
			res.append(char)
			stack = []
			for j in range(k-1):
				if len(res) == len(str): return "".join(res)
				if not heap: return ""
				fre, nex = heapq.heappop(heap)
				res.append(nex)
				if fre < -1: 
					stack.append((fre+1,nex))
			while stack:
				heapq.heappush(heap, stack.pop())
			heapq.heappush(heap, (freq+1,char))
		return "".join(res)

# 359 Logger Rate Limiter
# Design a logger system that receive stream of messages along with its timestamps, each message should be printed if and only if it is not printed in the last 10 seconds.
# Given a message and a timestamp (in seconds granularity), return true if the message should be printed in the given timestamp, otherwise returns false.
# It is possible that several messages arrive roughly at the same time.
class Logger:
	def __init__(self):
		self.ok = {}
	def shouldPrintMessage(self, timestamp, message):
		if timestamp < self.ok.get(message,0):
			return False
		self.ok[message] = timestamp+10
		return True

# 360 Sort Transformed Array
# Given a sorted array of integers nums and integer values a, b and c. Apply a function of the form f(x) = ax2 + bx + c to each element x in the array.
# The returned array must be in sorted order.
# Expected time complexity: O(n)
class Solution:
	def sortTransformedArray(self, nums, a, b, c):
		ret, l, r = [], 0, len(nums)-1
		while l <= r:
			n_l = a*nums[l]**2 + b*nums[l] + c
			n_r = a*nums[r]**2 + b*nums[r] + c
			if a >= 0:
				if n_l >= n_r:
					ret.append(n_l); l += 1
				else:
					ret.append(n_r); r -= 1
			else:
				if n_l < n_r:
					ret.append(n_l); l += 1
				else:
					ret.append(n_r); r -= 1
		if a >= 0:
			return ret[::-1]
		return ret

# 361 Bomb Enemy
# Given a 2D grid, each cell is either a wall 'W', an enemy 'E' or empty '0' (the number zero), return the maximum enemies you can kill using one bomb.
# The bomb kills all the enemies in the same row and column from the planted point until it hits the wall since the wall is too strong to be destroyed.
# Note that you can only put the bomb at an empty cell.
class Solution:
	def maxKilledEnemies(self, grid):
		def hits(grid):
			return [[h for block in ''.join(row).split('W') for h in [block.count('E')]*len(block)+[0]] for row in grid]
		rowhits, colhits = hits(grid), zip(*hits(zip(*grid)))
		return max([rh + ch for row in zip(grid, rowhits, colhits)
							for cell, rh, ch in zip(*row)
							if cell == '0'] or [0])

# 362 Design Hit Counter
# Design a hit counter which counts the number of hits received in the past 5 minutes.
# Each function accepts a timestamp parameter (in seconds granularity) and you may assume that calls are being made to the system 
# in chronological order (ie, the timestamp is monotonically increasing). You may assume that the earliest timestamp starts at 1.
class HitCounter:
	from collections import deque
	def __init__(self):
		self.num_of_hits = 0
		self.time_hits = deque()
	def hit(self, timestamp):
		if not self.time_hits or self.time_hits[-1][0] != timestamp:
			self.time_hits.append([timestamp, 1])
		else:
			self.time_hits[-1][1] += 1
		self.num_of_hits += 1
	def getHits(self, timestamp):
		while self.time_hits and self.time_hits[0][0] <= timestamp-300:
			self.num_of_hits -= self.time_hits.popleft()[1]
		return self.num_of_hits

# 363 Max Sum of Rectangle No Larger Than K
# Given a non-empty 2D matrix matrix and an integer k, find the max sum of a rectangle in the matrix such that its sum is no larger than k.
class Solution:
	import bisect
	def maxSumSubmatrix(self, matrix, k):
		h, w = len(matrix), len(matrix[0])
		A, rst = [[0]*w for _ in range(h)], float('-inf')
		for i in range(h):
			A[i][0] = matrix[i][0]
			for j in range(1, w):
				A[i][j] = A[i][j-1] + matrix[i][j]
		for j in range(w):
			for s in range(j, w):
				x, t = [0], 0
				for r in range(h):
					t += A[r][s] if j == 0 else A[r][s]-A[r][j-1]
					u = bisect.bisect_left(x, t-k)
					if u <= r:
						if rst == k: return k
						rst = max(rst, t-x[u])
					bisect.insort(x, t)
		return rst

# 364 Nested List Weight Sum II
# Given a nested list of integers, return the sum of all integers in the list weighted by their depth.
# Each element is either an integer, or a list -- whose elements may also be integers or other lists.
# Different from the previous question where weight is increasing from root to leaf, now the weight is defined from bottom up. i.e., the leaf level integers have weight 1, and the root level integers have the largest weight.
class Solution:
	def depthSumInverse(self, nestedList):
		maxDeep = self.dfsDeep(nestedList)
		return self.dfsHelper(nestedList, 1, maxDeep)  
	def dfsDeep(self, nestedList):
		if len(nestedList) == 1 and nestedList[0].isInteger(): return 1
		deep = 0	
		for nInt in nestedList:
			if nInt.isInteger():
				deep = max(1, deep)
			else:
				deep = max(self.dfsDeep(nInt.getList())+1, deep)
		return deep
	def dfsHelper(self, nestedList, level, maxDeep):
		if len(nestedList) == 1 and nestedList[0].isInteger():
			return (maxDeep-level+1)*nestedList[0].getInteger()
		s = 0
		for nInt in nestedList:
			if nInt.isInteger():
				s += (maxDeep-level+1)*nInt.getInteger()
			else:
				s += self.dfsHelper(nInt.getList(), level+1, maxDeep)
		return s

# 365 Water and Jug Problem
# You are given two jugs with capacities x and y litres. There is an infinite amount of water supply available. You need to determine whether it is possible to measure exactly z litres using these two jugs.
# If z liters of water is measurable, you must have z liters of water contained within one or both buckets by the end.
# Operations allowed:
# Fill any of the jugs completely with water.
# Empty any of the jugs.
# Pour water from one jug into another till the other jug is completely full or the first jug itself is empty.
class Solution:
	from fractions import gcd
	def canMeasureWater(self, x, y, z):
		return z==0 or x+y>=z and z%gcd(x,y)==0

# 366 Find Leaves of Binary Tree
# Given a binary tree, find all leaves and then remove those leaves. Then repeat the previous steps until the tree is empty.
class Solution:
	def findLeaves(self, root):
		def order(root, dic):
			if not root: return 0
			left = order(root.left, dic)
			right = order(root.right, dic)
			lev = max(left,right)+1
			dic[lev] += root.val,
			return lev
		dic, ret = collections.defaultdict(list), []
		order(root, dic)
		for i in range(1, len(dic)+1):
			ret.append(dic[i])
		return ret

# 367 Valid Perfect Square
# Given a positive integer num, write a function which returns True if num is a perfect square else False.
class Solution:
	def isPerfectSquare(self, num):
		l, h = 1, num/2
		while l<h:
			mid = (l+h)/2
			if mid*mid==num: return True
			if mid*mid>num: h = mid-1
			else: l = mid+1
		return l*l==num

# 368 Largest Divisible Subset
# Given a set of distinct positive integers, find the largest subset such that every pair (Si, Sj) of elements in this subset satisfies: Si % Sj = 0 or Sj % Si = 0.
# If there are multiple solutions, return any subset is fine.
class Solution:
	def largestDivisibleSubset(self, nums):
		S = {-1: set()}
		for x in sorted(nums):
			S[x] = max((S[d] for d in S if x%d == 0), key=len)|{x}
		return list(max(S.values(), key=len))

# 369 Plus One Linked List
# Given a non-negative number represented as a singly linked list of digits, plus one to the number.
# The digits are stored such that the most significant digit is at the head of the list.
class Solution:
	def plusOne(self, head):
		tail = None
		while head:
			head.next, head, tail = tail, head.next, head
		carry = 1
		while tail:
			carry, tail.val = divmod(carry+tail.val, 10)
			if carry and not tail.next:
				tail.next = ListNode(0)
			tail.next, tail, head = head, tail.next, tail
		return head

# 370 Range Addition
# Assume you have an array of length n initialized with all 0's and are given k update operations.
# Each operation is represented as a triplet: [startIndex, endIndex, inc] which increments each element of subarray A[startIndex ... endIndex] (startIndex and endIndex inclusive) with inc.
# Return the modified array after all k operations were executed.
class Solution:
	def getModifiedArray(self, length, updates):
		res = [0]*length
		for update in updates:
			start, end, inc = update
			res[start] += inc
			if end+1 <= length-1:
				res[end+1] -= inc
		sum = 0
		for i in range(length):
			sum += res[i]
			res[i] = sum
		return res

# 371 Sum of Two Integers
# Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.
class Solution:
	def getSum(self, a, b):
		MAX = 0x7FFFFFFF
		MIN = 0x80000000
		mask = 0xFFFFFFFF
		while b != 0:
			# ^ get different bits and & gets double 1s, << moves carry
			a, b = (a^b)&mask, ((a&b)<<1)&mask
		# if a is negative, get a's 32 bits complement positive first
		# then get 32-bit positive's Python complement negative
		return a if a<=MAX else ~(a^mask)

# 372 Super Pow
# Your task is to calculate ab mod 1337 where a is a positive integer and b is an extremely large positive integer given in the form of an array.
# https://discuss.leetcode.com/topic/50591/fermat-and-chinese-remainder/2
class Solution:
	def superPow(self, a, b):
		def mod(p):
			return pow(a, reduce(lambda e,d:(10*e+d)%(p-1),b,0), p) if a%p else 0
		return (764*mod(7)+574*mod(191))%1337

# 373 Find K Pairs with Smallest Sums
# You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.
# Define a pair (u,v) which consists of one element from the first array and one element from the second array.
# Find the k pairs (u1,v1),(u2,v2) ...(uk,vk) with the smallest sums.
class Solution:
	def kSmallestPairs(self, nums1, nums2, k):
		pairs = []; queue = []
		def push(i, j):
			if i < len(nums1) and j < len(nums2):
				heapq.heappush(queue, [nums1[i]+nums2[j], i, j])
		push(0, 0)
		while queue and len(pairs) < k:
			_, i, j = heapq.heappop(queue)
			pairs.append([nums1[i], nums2[j]])
			push(i, j+1)
			if j == 0:
				push(i+1, 0)
		return pairs

# 374 Guess Number Higher or Lower
# We are playing the Guess Game. The game is as follows:
# I pick a number from 1 to n. You have to guess which number I picked.
# Every time you guess wrong, I'll tell you whether the number is higher or lower.
# You call a pre-defined API guess(int num) which returns 3 possible results (-1, 1, or 0):
class Solution:
	def guessNumber(self, n):
		lo, hi = 1, n
		while lo < hi:
			mid = (lo+hi)/2
			if guess(mid) == 1:
				lo = mid+1
			else:
				hi = mid
		return lo

# 375 Guess Number Higher or Lower II
# However, when you guess a particular number x, and you guess wrong, you pay $x. You win the game when you guess the number I picked.
# Given a particular n ≥ 1, find out how much money you need to have to guarantee a win.
class Solution:
	def getMoneyAmount(self, n):
		need = [[0]*(n+1) for _ in range(n+1)]
		for lo in range(n, 0, -1):
			for hi in range(lo+1, n+1):
				need[lo][hi] = min(x+max(need[lo][x-1],need[x+1][hi]) for x in range(lo, hi))
		return need[1][n]

# 376 Wiggle Subsequence
# For example, [1,7,4,9,2,5] is a wiggle sequence because the differences (6,-3,5,-7,3) are alternately positive and negative.
class Solution:
	def wijggleMaxLength(self, nums):
		mono, ans = None, 1
		for i in xrange(1, len(nums)):
			diff = nums[i] - nums[i-1]
			if diff == 0: continue
			if (not mono) or (mono and diff*mono<0):
				ans += 1
				mono = diff
		return ans if len(nums) else 0

# 377 Combination Sum IV
# Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.
class Solution:
	def combinationSum4(self, nums, target):
		nums, combs = sorted(nums), [1]+[0]*(target)
		for i in range(target+1):
			for num in nums:
				if num  > i: break
				if num == i: combs[i] += 1
				if num  < i: combs[i] += combs[i-num]
		return combs[target]

# 378 Kth Smallest Element in a Sorted Matrix
# Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.
# Note that it is the kth smallest element in the sorted order, not the kth distinct element.
class Solution:
	def kthSmallest(self, matrix, k):
		heap, res, n = [(matrix[0][0], 0, 0)], 0, len(matrix)
		for k in range(1, k+1):
			res, row, col = heapq.heappop(heap)
			if not row and col < n-1:
				heapq.heappush(heap, (matrix[row][col+1], row, col+1))
			if row < n-1:
				heapq.heappush(heap, (matrix[row+1][col], row+1, col))
		return res

# 379 Design Phone Directory
# Design a Phone Directory which supports the following operations:
# get: Provide a number which is not assigned to anyone.
# check: Check if a number is available or not.
# release: Recycle or release a number.
class PhoneDirectory:
	def __init__(self, maxNumbers):
		self.available = set(range(maxNumbers))
	def get(self):
		return self.available.pop() if self.available else -1
	def check(self, number):
		return number in self.available
	def release(self, number):
		self.available.add(number)

# 380 Insert Delete GetRandom O(1)
# Design a data structure that supports all following operations in average O(1) time.
# insert(val): Inserts an item val to the set if not already present.
# remove(val): Removes an item val from the set if present.
# getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.
class RandomizedSet:
	import random
	def __init__(self):
		self.nums, self.pos = [], {}
	def insert(self, val):
		if val not in self.pos:
			self.nums.append(val)
			self.pos[val] = len(self.nums)-1
			return True
		return False
	def remove(self, val):
		if val in self.pos:
			idx, last = self.pos[val], self.nums[-1]
			self.nums[idx], self.pos[last] = last, idx
			self.nums.pop(); self.pos.pop(val, 0)
			return True
		return False
	def getRandom(self):
		return self.nums[random.randint(0,len(self.nums)-1)]

# 381 Insert Delete GetRandom O(1) - Duplicates allowed
class RandomizedCollection:
	import random
	def __init__(self):
		self.vals, self.idxs = [], collections.defaultdict(set)
	def insert(self, val):
		self.vals.append(val)
		self.idxs[val].add(len(self.vals)-1)
		return len(self.idxs[val]) == 1
	def remove(self, val):
		if self.idxs[val]:
			out, ins = self.idxs[val].pop(), self.vals[-1]
			self.vals[out] = ins
			if self.idxs[ins]:
				self.idxs[ins].add(out)
				self.idxs[ins].discard(len(self.vals)-1)
			self.vals.pop()
			return True
		return False 
	def getRandom(self):
		return random.choice(self.vals)

# 382 Linked List Random Node
# Given a singly linked list, return a random node's value from the linked list. Each node must have the same probability of being chosen.
class Solution:
	import random
	def __init__(self, head):
		self.head = head
	def getRandom(self):
		curr = self.head
		selected_node, max_val = None, 0
		while curr is not None:
			val = random.random()
			if val > max_val:
				selected_node, max_val = curr, val
			curr = curr.next
		return selected_node.val

# 383 Ransom Note
#  Given  an  arbitrary  ransom  note  string  and  another  string  containing  letters from  all  the  magazines,  
# write  a  function  that  will  return  true  if  the  ransom  note  can  be  constructed  from  the  magazines ;  otherwise,  it  will  return  false.   
class Solution:
	def canConstruct(self, ransomNote, magazine):
		return not collections.Counter(ransomNote) - collections.Counter(magazine)

# 384 Shuffle an Array
# Shuffle a set of numbers without duplicates.
class Solution:
	import random
	def __init__(self, nums):
		self.nums = nums
	def reset(self):
		return self.nums
	def shuffle(self):
		nums = [i for i in range(len(self.nums))]
		for i in range(len(self.nums)-1, -1, -1):
			# choose a number to put at i
			index = random.randint(0, i)
			nums[i], nums[index] = nums[index], nums[i]
		return [self.nums[p] for p in nums]

# 385 Mini Parser
# Given a nested list of integers represented as a string, implement a parser to deserialize it.
# Each element is either an integer, or a list -- whose elements may also be integers or other lists.
# Note: You may assume that the string is well-formed:
# String is non-empty.
# String does not contain white spaces.
# String contains only digits 0-9, [, - ,, ].
class Solution:
	def deserialize(self, s):
		if s[0] != '[':
			return NestedInteger(int(s))
		ret, l = NestedInteger(), [] if s=='[]' else s[1:-1].split(',')
		cnt, lo = 0, 0
		for hi, ss in enumerate(l):
			cnt += ss.count('[') - ss.count(']')
			if cnt == 0:
				ret.add(self.deserialize(','.join(l[lo:hi+1])))
				lo = hi+1
		return ret

# 386 Lexicographical Numbers
# Given an integer n, return 1 - n in lexicographical order.
# For example, given 13, return: [1,10,11,12,13,2,3,4,5,6,7,8,9].
# Please optimize your algorithm to use less time and space. The input size may be as large as 5,000,000.
class Solution:
	def lexicalOrder(self, n):
		def dfs(start,n,res):
			for i in xrange(start,start+10):
				if i<=n and len(res)<n:
					res.append(i)
					if i*10<=n: 
						dfs(i*10,n,res)
		res = []
		dfs(1,n,res)
		return res

# 387 First Unique Character in a String
# Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.
class Solution:
	def firstUniqChar(self, s):
		l =[0]*256
		for i in range(0,len(s)):
			l[ord(s[i])] += 1 
		for i in range(0,len(s)):
			if l[ord(s[i])] == 1:
				return i
		return -1

# 388 Longest Absolute File Path
# Suppose we abstract our file system by a string in the following manner:
# The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:
# dir
#	 subdir1
#	 subdir2
#		 file.ext
# We are interested in finding the longest (number of characters) absolute path to a file within our file system. 
# For example, in the second example above, the longest absolute path is "dir/subdir2/subsubdir2/file2.ext", and its length is 32 (not including the double quotes).
class Solution:
	def lengthLongestPath(self, input):
		currlen, maxlen = 0, 0
		stack = []
		for s in input.split('\n'):
			depth = s.count('\t')
			while len(stack) > depth:
				currlen -= stack.pop()
			stack.append(len(s.strip('\t'))+1)
			currlen += stack[-1]
			if '.' in s:
				maxlen = max(maxlen, currlen-1)
		return maxlen

# 389 Find the Difference
# Given two strings s and t which consist of only lowercase letters.
# String t is generated by random shuffling string s and then add one more letter at a random position.
# Find the letter that was added in t.
class Solution:
	def findTheDifference(self, s, t):
		s, t = sorted(s), sorted(t)
		return t[-1] if s == t[:-1] else [x[1] for x in zip(s, t) if x[0] != x[1]][0]

# 390 Elimination Game
class Solution:
	def lastRemaining(self, n):
		nums = range(1, n+1)
		while len(nums) > 1:
			nums = nums[1::2][::-1]
		return nums[0]

# 391 Perfect Rectangle
class Solution(object):
	def isRectangleCover(self, rectangles):
		def recordCorner(point):
			if point in corners:
				corners[point] += 1
			else:
				corners[point] = 1
		corners = {} # record all corners 
		L, B, R, T, area = float('inf'), float('inf'), -float('inf'), -float('inf'), 0
		for sub in rectangles:
			L, B, R, T = min(L, sub[0]), min(B, sub[1]), max(R, sub[2]), max(T, sub[3])
			ax, ay, bx, by = sub[:]
			area += (bx-ax)*(by-ay) # sum up the area of each sub-rectangle
			map(recordCorner, [(ax, ay), (bx, by), (ax, by), (bx, ay)])
		if area != (T-B)*(R-L): return False # check the area
		big_four = [(L,B),(R,T),(L,T),(R,B)]
		for bf in big_four: # check corners of big rectangle
			if bf not in corners or corners[bf] != 1:
				return False
		for key in corners: # check existing "inner" points
			if corners[key]%2 and key not in big_four:
				return False
		return True

# 392 Is Subsequence
# Given a string s and a string t, check if s is subsequence of t.
class Solution:
	def isSubsequence(self, s, t):
		if len(s) == 0: return True
		if len(t) == 0: return False 
		i, j = 0, 0
		while i < len(s) and j < len(t):
			if s[i] == t[j]: i += 1
			j += 1
		return i == len(s)

# 393 UTF-8 Validation
class Solution:
	def check(nums, start, size):
		for i in range(start+1, start+size+1):
			if i>=len(nums) or (nums[i]>>6) != 0b10: return False
		return True
	def validUtf8(self, nums, start=0):
		while start < len(nums):
			first = nums[start]
			if   (first>>3) == 0b11110 and check(nums, start, 3): start += 4
			elif (first>>4) == 0b1110 and check(nums, start, 2): start += 3
			elif (first>>5) == 0b110 and check(nums, start, 1): start += 2
			elif (first>>7) == 0: start += 1
			else: return False
		return True

# 394 Decode String
# Given an encoded string, return it's decoded string.
# The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times.
# s = "3[a]2[bc]", return "aaabcbc".
# s = "3[a2[c]]", return "accaccacc".
# s = "2[abc]3[cd]ef", return "abcabccdcdcdef".
class Solution:
	def decodeString(self, s):
		stack = [["", 1]]; num = ""
		for ch in s:
			if ch.isdigit():
			  num += ch
			elif ch == '[':
				stack.append(["", int(num)]); num = ""
			elif ch == ']':
				st, k = stack.pop()
				stack[-1][0] += st*k
			else:
				stack[-1][0] += ch
		return stack[0][0]

# 395 Longest Substring with At Least K Repeating Characters
# Find the length of the longest substring T of a given string (consists of lowercase letters only) such that every character in T appears no less than k times.
class Solution:
	def longestSubstring(self, s, k):
		if len(s) < k: return 0
		c = min(set(s), key=s.count)
		if s.count(c) >= k: return len(s)
		return max(self.longestSubstring(t, k) for t in s.split(c))

# 396 Rotate Function
# Given an array of integers A and let n to be its length.
# Assume Bk to be an array obtained by rotating the array A k positions clock-wise, we define a "rotation function" F on A as follow:
# F(k) = 0 * Bk[0] + 1 * Bk[1] + ... + (n-1) * Bk[n-1]. Calculate the maximum value of F(0), F(1), ..., F(n-1).
# A = [4, 3, 2, 6]
# F(0) = (0 * 4) + (1 * 3) + (2 * 2) + (3 * 6) = 0 + 3 + 4 + 18 = 25
# F(1) = (0 * 6) + (1 * 4) + (2 * 3) + (3 * 2) = 0 + 4 + 6 + 6 = 16
# F(2) = (0 * 2) + (1 * 6) + (2 * 4) + (3 * 3) = 0 + 6 + 8 + 9 = 23
# F(3) = (0 * 3) + (1 * 2) + (2 * 6) + (3 * 4) = 0 + 2 + 12 + 12 = 26
# So the maximum value of F(0), F(1), F(2), F(3) is F(3) = 26.
class Solution:
	def maxRotateFunction(self, A):
		if len(A) == 0: return 0
		totalSum = sum(A); lMax = 0 
		for i in range(len(A)):
			lMax += i*A[i]
		gMax = lMax
		for i in range(len(A)-1, 0, -1):
			lMax += totalSum-A[i]*len(A)
			gMax = max(gMax, lMax)
		return gMax

# 397 Integer Replacement
# Given a positive integer n and you can do operations as follow:
# If n is even, replace n with n/2.
# If n is odd, you can replace n with either n + 1 or n - 1.
# What is the minimum number of replacements needed for n to become 1?
class Solution:
	def integerReplacement(self, n):
		if n == 1: return 0
		if n % 2:
			return 1+min(self.integerReplacement(n+1), self.integerReplacement(n-1))
		else:
			return 1+self.integerReplacement(n/2)

# 398 Random Pick Index
# Given an array of integers with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.
class Solution:
	def __init__(self, nums):
		self.indexes = {}
		for i, num in enumerate(nums):
			I = self.indexes.get(num)
			if I is None:
				self.indexes[num] = i
			elif isinstance(I, int):
				self.indexes[num] = [I, i]
			else:
				self.indexes[num].append(i)
	def pick(self, target):
		I = self.indexes[target]
		return I if isinstance(I, int) else random.choice(I)

# 399 Evaluate Division
# Equations are given in the format A / B = k, where A and B are variables represented as strings, and k is a real number (floating point number). Given some queries, return the answers. If the answer does not exist, return -1.0.
# Given a / b = 2.0, b / c = 3.0. 
# queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? . 
# return [6.0, 0.5, -1.0, 1.0, -1.0 ].
class Solution:
	def calcEquation(self, equations, values, queries):
		quot = collections.defaultdict(dict)
		for (num, den), val in zip(equations, values):
			quot[num][num] = quot[den][den] = 1.0
			quot[num][den] = val
			quot[den][num] = 1 / val
		for k in quot:
			for i in quot[k]:
				for j in quot[k]:
					quot[i][j] = quot[i][k] * quot[k][j]
		return [quot[num].get(den, -1.0) for num, den in queries]

# 400 Nth Digit
# Find the nth digit of the infinite integer sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...
def findNthDigit(self, n):
	n -= 1
	for digits in range(1, 11):
		first = 10**(digits-1)
		if n < 9*first*digits:
			return int(str(first+n/digits)[n%digits])
		n -= 9*first*digits

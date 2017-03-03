# -*- coding: utf-8 -*- 

# 201 Bitwise AND of Numbers Range 
# Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.
# For example, given the range [5, 7], you should return 4.
class Solution:
	def rangeBitwiseAnd(self, m, n):
		i, diff = 0, n-m
		while diff:
			diff >>= 1
			i += 1
		return n&m >> i << i

# 202 Happy Number
# Write an algorithm to determine if a number is "happy".
# A happy number is a number defined by the following process: Starting with any positive integer, 
# replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), 
# or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.
# Example: 19 is a happy number
# 1**2 + 9**2 = 82
# 8**2 + 2**2 = 68
# 6**2 + 8**2 = 100
# 1**2 + 0**2 + 0**2 = 1
class Solution:
	def isHappy(self, n):
		lookup = {}
		while n != 1 and n not in lookup:
			lookup[n] = True
			n = self.nextNumber(n)
		return n == 1
	def nextNumber(self, n):
		new = 0
		for char in str(n):
			new += int(char)**2
		return new

# 203 Remove Linked List Elements
# Remove all elements from a linked list of integers that have value val.
class Solution:
	def removeElements(self, head, val):
		dummy = ListNode(float("-inf"))
		dummy.next = head
		prev, curr = dummy, dummy.next
		while curr:
			if curr.val == val:
				prev.next = curr.next
			else:
				prev = curr
			curr = curr.next
		return dummy.next

# 204 Count Primes
# Count the number of prime numbers less than a non-negative number, n
class Solution:
	def countPrimes(self, n):
		if n <= 2: return 0
		is_prime, num = [True]*n, 0
		for i in xrange(2, n):
			if is_prime[i]:
				num += 1
				for j in xrange(i+i, n, i):
					is_prime[j] = False
		return num

# 205 Isomorphic Strings
# Given two strings s and t, determine if they are isomorphic.
# Two strings are isomorphic if the characters in s can be replaced to get t.
class Solution:
	def isIsomorphic(self, s, t):
		sourceMap, targetMap = {}, {}
		for x in range(len(s)):
			source, target = sourceMap.get(t[x]), targetMap.get(s[x])
			if source is None and target is None:
				sourceMap[t[x]], targetMap[s[x]] = s[x], t[x]
			elif target != t[x] or source != s[x]:
				return False
		return True

# 206 Reverse Linked List
class Solution:
	def reverseList(self, head):
		dummy = ListNode(0)
		while head:
			next = head.next
			head.next = dummy.next
			dummy.next = head
			head = next
		return dummy.next

# 207 Course Schedule
# There are a total of n courses you have to take, labeled from 0 to n - 1.
# Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]
# Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?
class Solution:
	def canFinish(self, numCourses, prerequisites):
		degrees = [0]*numCourses
		childs = [[] for x in range(numCourses)]
		for pair in prerequisites:
			degrees[pair[0]] += 1
			childs[pair[1]].append(pair[0])
		courses = set(range(numCourses))
		flag = True
		while flag and len(courses):
			flag = False
			removeList = []
			for x in courses:
				if degrees[x] == 0:
					for child in childs[x]:
						degrees[child] -= 1
					removeList.append(x)
					flag = True
			for x in removeList:
				courses.remove(x)
		return len(courses) == 0

# 208 Implement Trie (Prefix Tree)
# Implement a trie with insert, search, and startsWith methods.
class TrieNode:
	def __init__(self):
		self.childs = {}
		self.isWord = False
class Trie:
	def __init__(self):
		self.root = TrieNode()
	def insert(self, word):
		node = self.root
		for letter in word:
			child = node.childs.get(letter)
			if child is None:
				child = TrieNode()
				node.childs[letter] = child
			node = child
		node.isWord = True
	def search(self, word):
		node = self.root
		for letter in word:
			node = node.childs.get(letter)
			if node is None:
				return False
		return node.isWord
	def startsWith(self, prefix):
		node = self.root
		for letter in prefix:
			node = node.childs.get(letter)
			if node is None:
				return False
		return True

# 209 Minimum Size Subarray Sum
# Given an array of n positive integers and a positive integer s, find the minimal length of a subarray of which the sum ≥ s. If there isn't one, return 0 instead.
class Solution:
	def minSubArrayLen(self, s, nums):
		if sum(nums) < s: return 0
		tail, total, minlen = 0, 0, len(nums)
		for head in xrange(0,len(nums)):
			total = total+nums[head]
			while total >= s:
				minlen = min(minlen, head-tail+1)
				total = total-nums[tail]
				tail += 1
		return minlen

# 210 Course Schedule II
# There are a total of n courses you have to take, labeled from 0 to n - 1.
# Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]
# Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.
# There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.
class Solution:
	def findOrder(self, numCourses, prerequisites):
		degrees = [0] * numCourses
		childs = [[] for x in range(numCourses)]
		for pair in prerequisites:
			degrees[pair[0]] += 1
			childs[pair[1]].append(pair[0])
		courses = set(range(numCourses))
		ans, flag = [], True
		while flag and len(courses):
			flag = False
			removeList = []
			for x in courses:
				if degrees[x] == 0:
					for child in childs[x]:
						degrees[child] -= 1
					removeList.append(x)
					flag = True
			for x in removeList:
				ans.append(x)
				courses.remove(x)
		return [[], ans][len(courses) == 0]

# 211 Add and Search Word - Data structure design
# Design a data structure that supports the following two operations:
# void addWord(word)
# bool search(word)
# search(word) can search a literal word or a regular expression string containing only letters a-z or .. A . means it can represent any one letter.
class TrieNode:
	def __init__(self):
		self.childs = {}
		self.isWord = False
class WordDictionary:
	def __init__(self):
		self.root = TrieNode()
	def addWord(self, word):
		node = self.root
		for letter in word:
			child = node.childs.get(letter)
			if child is None:
				child = TrieNode()
				node.childs[letter] = child
			node = child
		node.isWord = True
	def search(self, word):
		return self.find(self.root, word)
	def find(self, node, word):
		if word == '':
			return node.isWord
		if word[0] == '.':
			for x in node.childs:
				if self.find(node.childs[x], word[1:]):
					return True
		else:
			child = node.childs.get(word[0])
			if child:
				return self.find(child, word[1:])
		return False

# 212 Word Search II
# Given a 2D board and a list of words from the dictionary, find all words in the board.
# Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring.
# The same letter cell may not be used more than once in a word.
class TrieNode:
	def __init__(self):
		self.word = None
		self.children = {}
class Trie:
	def __init__(self):
		self.root = TrieNode()
	def insert(self, word):
		root = self.root
		for char in word:
			root = root.children.setdefault(char, TrieNode())
		root.word = word
class Solution:
	def search(self, i, j, root, board, m, n, r):
		char = board[i][j]
		if not (char and char in root.children): return
		board[i][j], root = None, root.children[char]
		if root.word:
			r.append(root.word)
			root.word = None
		for x, y in ((0, -1), (-1, 0), (0, 1), (1, 0)):
			ii, jj = i+x, j+y
			if 0 <= ii < m and 0 <= jj < n:
				self.search(ii, jj, root, board, m, n, r)
		board[i][j] = char
	def findWords(self, board, words):
		if not board: return []
		tree = Trie()
		[tree.insert(word) for word in words]
		m, n, r = len(board), len(board[0]), []
		for i, row in enumerate(board):
			for j, char in enumerate(row):
				self.search(i, j, tree.root, board, m, n, r)
		return r

# 213 House Robber II
# After robbing those houses on that street, the thief has found himself a new place for his thievery so that he will not get too much attention. 
# This time, all houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. 
# Meanwhile, the security system for these houses remain the same as for those in the previous street.
class Solution:
	def rob(self, nums):
		if len(nums) == 1: return nums[0]
		return max(self.robLinear(nums[1:]), self.robLinear(nums[:-1]))
	def robLinear(self, num):
		size, odd, even = len(num), 0, 0
		for i in range(size):
			if i % 2:
				odd = max(odd+num[i], even)
			else:
				even = max(even+num[i], odd)
		return max(odd, even)

# 214 Shortest Palindrome
# Given a string S, you are allowed to convert it to a palindrome by adding characters in front of it. 
# Find and return the shortest palindrome you can find by performing this transformation.
class Solution:
	def shortestPalindrome(self, s):
		r = s[::-1]
		for i in range(len(s)+1):
			if s.startswith(r[i:]):
				return r[:i]+s

# 215 Kth Largest Element in an Array
import random
class Solution:
	def findKthLargest(self, nums, k):
		pivot = random.choice(nums)
		nums1, nums2 = [], []
		for num in nums:
			if num > pivot:
				nums1.append(num)
			elif num < pivot:
				nums2.append(num)
		if k <= len(nums1):
			return self.findKthLargest(nums1, k)
		if k > len(nums) - len(nums2):
			return self.findKthLargest(nums2, k-(len(nums)-len(nums2)))
		return pivot

# 216 Combination Sum III
# Find all possible combinations of k numbers that add up to a number n, 
# given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.
# Ensure that numbers within the set are sorted in ascending order.
class Solution:
	def combinationSum3(self, k, n):
		ans = []
		def search(start, cnt, sums, nums):
			if cnt > k or sums > n: return
			if cnt == k and sums == n:
				ans.append(nums); return
			for x in range(start+1, 10):
				search(x, cnt+1, sums+x, nums+[x])
		search(0, 0, 0, [])
		return ans

# 217 Contains Duplicate
# Given an array of integers, find if the array contains any duplicates. 
class Solution:
	def containsDuplicate(self, nums):
		return len(nums) != len(set(nums))

# 218 The Skyline Problem
# A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance.
# Now suppose you are given the locations and height of all the buildings as shown on a cityscape photo, write a program to output the skyline formed by these buildings collectively.
class Solution:
	def getSkyline(self, buildings):
		Xs = []
		for l, r, _ in buildings:
			Xs.extend([l,r])
		Xs.sort()
		# maxheap to store heights and rights
		result, heap, i = [], [], 0
		for x in Xs:
			# remove ended max buildings to reveal current max building
			while heap and heap[0][1] <= x:
				heapq.heappop(heap)
			# add starting buildings
			while i < len(buildings) and buildings[i][0] == x:
				heapq.heappush(heap, (-buildings[i][2], buildings[i][1]))
				i += 1
			# y is either 0 or max height
			y = 0 if not heap else -heap[0][0]
			# if y differs
			if not result or result[-1][1] != y:
				result.append([x, y])
		return result

# 219 Contains Duplicate II
# Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the difference between i and j is at most k.
class Solution:
	def containsNearbyDuplicate(self, nums, k):
		numDict = {}
		for x in range(len(nums)):
			idx = numDict.get(nums[x])
			if idx >= 0 and x-idx <= k:
				return True
			numDict[nums[x]] = x
		return False

# 220 Contains Duplicate III
# Given an array of integers, find out whether there are two distinct indices i and j in the array such that the difference between nums[i] and nums[j] is at most t and the difference between i and j is at most k.
class Solution:
	def containsNearbyAlmostDuplicate(self, nums, k, t):
		d = {}
		for i, n in enumerate(nums):
			bucket, offset = (n/t, 1) if t != 0 else (n, 0)
			for b in range(bucket-offset, bucket+offset+1):
				if b in d and abs(i-d[b][0]) <= k and abs(n-d[b][1]) <= t:
					return True
			d[bucket] = (i, n)
		return False

# 221 Maximal Square
# Given a 2D binary matrix filled with 0's and 1's, find the largest square containing all 1's and return its area.
class Solution:
	def maximalSquare(self, matrix):
		if matrix == []: return 0
		m, n = len(matrix), len(matrix[0])
		ans, dp = 0, [[0 for y in xrange(n)] for x in range(m)]
		for x in range(m):
			for y in range(n):
				dp[x][y] = int(matrix[x][y])
				if x and y and dp[x][y]:
					dp[x][y] = min(dp[x-1][y-1], dp[x][y-1], dp[x-1][y])+1
				ans = max(ans, dp[x][y])
		return ans*ans

# 222 Count Complete Tree Nodes
# Given a complete binary tree, count the number of nodes.
class Solution:
	def countNodes(self, root):
		h, nodes = self.height(root), 0
		while root:
			if self.height(root.right) == h-1:
				nodes += 2**h
				root = root.right
			else:
				nodes += 2**(h-1)
				root = root.left
			h -= 1
		return nodes
	def height(self, root):
		return -1 if not root else 1 + self.height(root.left)

# 223 Rectangle Area
# Find the total area covered by two rectilinear rectangles in a 2D plane.
# Each rectangle is defined by its bottom left corner and top right corner as shown in the figure.
class Solution:
	def computeArea(self, A, B, C, D, E, F, G, H):
		sums = (C-A)*(D-B)+(G-E)*(H-F)
		return sums-max(min(C,G)-max(A,E),0)*max(min(D,H)-max(B,F),0)

# 224 Basic Calculator
# Implement a basic calculator to evaluate a simple expression string.
# The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces 
class Solution:
	def calculate(self, s):
		res, num, sign, stack = 0, 0, 1, []
		for ss in s:
			if ss.isdigit():
				num = 10*num + int(ss)
			elif ss in ["-", "+"]:
				res += sign*num; num = 0
				sign = [-1, 1][ss=="+"]
			elif ss == "(":
				stack.append(res)
				stack.append(sign)
				sign, res = 1, 0
			elif ss == ")":
				res += sign*num
				res *= stack.pop()
				res += stack.pop()
				num = 0
		return res + num*sign

# 225 Implement Stack using Queues
# Implement the following operations of a stack using queues.
# push(x) -- Push element x onto stack.
# pop() -- Removes the element on top of the stack.
# top() -- Get the top element.
# empty() -- Return whether the stack is empty.
class Stack:
	def __init__(self):
		self.queue = []
	def push(self, x):
		self.queue.append(x)
	def pop(self):
		for x in range(len(self.queue)-1):
			self.queue.append(self.queue.pop(0))
		self.queue.pop(0)
	def top(self):
		top = None
		for x in range(len(self.queue)):
			top = self.queue.pop(0)
			self.queue.append(top)
		return top
	def empty(self):
		return self.queue == []

# 226 Invert Binary Tree
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def invertTree(self, root):
		if root is None:
			return None
		if root.left:
			self.invertTree(root.left)
		if root.right:
			self.invertTree(root.right)
		root.left, root.right = root.right, root.left
		return root

# 227 Basic Calculator II
# Implement a basic calculator to evaluate a simple expression string.
# The expression string contains only non-negative integers, +, -, *, / operators and empty spaces. The integer division should truncate toward zero.
class Solution:
	def calculate(self, s):
		if not s:
			return "0"
		stack, num, sign = [], 0, "+"
		for i in xrange(len(s)):
			if s[i].isdigit():
				num = num*10+int(s[i])
			if (not s[i].isdigit() and not s[i].isspace()) or i == len(s)-1:
				if sign == "-":
					stack.append(-num)
				elif sign == "+":
					stack.append(num)
				elif sign == "*":
					stack.append(stack.pop()*num)
				else:
					tmp = stack.pop()
					if tmp//num < 0 and tmp%num != 0:
						stack.append(tmp//num+1)
					else:
						stack.append(tmp//num)
				sign = s[i]; num = 0
		return sum(stack)

# 228 Summary Ranges
# Given a sorted integer array without duplicates, return the summary of its ranges.
# For example, given [0,1,2,4,5,7], return ["0->2","4->5","7"].
class Solution:
	def summaryRanges(self, nums):
		ranges = []
		for n in nums:
			if not ranges or n > ranges[-1][-1]+1:
				ranges += [],
			ranges[-1][1:] = n,
		return ['->'.join(map(str, r)) for r in ranges]

# 229 Majority Element II
# Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.
# The algorithm should run in linear time and in O(1) space.
class Solution:
	def majorityElement(self, nums):
		n1 = n2 = None
		c1 = c2 = 0
		for num in nums:
			if n1 == num:
				c1 += 1
			elif n2 == num:
				c2 += 1
			elif c1 == 0:
				n1, c1 = num, 1
			elif c2 == 0:
				n2, c2 = num, 1
			else:
				c1, c2 = c1-1, c2-1
		size = len(nums)
		return [n for n in (n1, n2) if n is not None and nums.count(n) > size/3]

# 230 Kth Smallest Element in a BST
# Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def kthSmallest(self, root, k):
		result, stack, x = [], [], 0
		while stack or root:
			if root:
				stack.append(root)
				root = root.left
			else:
				root = stack.pop()
				x += 1
				if x == k:
					return root.val
				root = root.right

# 231 Power of Two
# Given an integer, write a function to determine if it is a power of two.
class Solution:
	def isPowerOfTwo(self, n):
		return n > 0 and n&(n-1) == 0

# 232 Implement Queue using Stacks
# Implement the following operations of a queue using stacks.
# push(x) -- Push element x to the back of queue.
# pop() -- Removes the element from in front of queue.
# peek() -- Get the front element.
# empty() -- Return whether the queue is empty.
class Queue:
	def __init__(self):
		self.stack = []
	def push(self, x):
		swap = []
		while self.stack:
			swap.append(self.stack.pop())
		swap.append(x)
		while swap:
			self.stack.append(swap.pop())
	def pop(self):
		self.stack.pop()
	def peek(self):
		return self.stack[-1]
	def empty(self):
		return len(self.stack) == 0

# 233 Number of Digit One
# Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than or equal to n.
# if n = xyzdabc 计算千位上1个总个数
# (1) xyz * 1000			if d == 0
# (2) xyz * 1000 + abc + 1	if d == 1
# (3) xyz * 1000 + 1000		if d > 1
class Solution:
	def countDigitOne(self, n):
		if n <= 0: return 0
		q, x, ans = n, 1, 0
		while q > 0:
			digit = q % 10
			q /= 10
			ans += q * x
			if digit == 1:
				ans += n%x+1
			elif digit > 1:
				ans += x
			x *= 10
		return ans

# 234 Palindrome Linked List
# Given a singly linked list, determine if it is a palindrome.
class ListNode:
	 def __init__(self, x):
		self.val = x
		self.next = None
class Solution:
	def isPalindrome(self, head):
		if head is None:
			return True
		fast = slow = head
		while fast.next and fast.next.next:
			slow = slow.next
			fast = fast.next.next
		p, last = slow.next, None
		while p:
			next = p.next
			p.next = last
			last, p = p, next
		p1, p2 = head, last 
		while p1 and p1.val == p2.val:
			p1, p2 = p1.next, p2.next
		#resume linked list(optional)
		p, last = last, None
		while p:
			next = p.next
			p.next = last
			last, p = p, next
		slow.next = last
		return p1 is None

# 235 Lowest Common Ancestor of a Binary Search Tree
# Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def lowestCommonAncestor(self, root, p, q):
		if (p.val-root.val)*(q.val-root.val) <= 0:
			return root
		elif p.val < root.val:
			return self.lowestCommonAncestor(root.left, p, q)
		else:
			return self.lowestCommonAncestor(root.right, p, q)

# 236 Lowest Common Ancestor of a Binary Tree
# Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
class Solution:
	def lowestCommonAncestor(self, root, p, q):
		stack, parent = [root], {root: None}
		while p not in parent or q not in parent:
			node = stack.pop()
			if node.left:
				parent[node.left] = node
				stack.append(node.left)
			if node.right:
				parent[node.right] = node
				stack.append(node.right)
		ancestors = set()
		while p:
			ancestors.add(p)
			p = parent[p]
		while q not in ancestors:
			q = parent[q]
		return q

# 237 Delete Node in a Linked List
# Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.
class ListNode:
	def __init__(self, x):
		self.val = x
		self.next = None
class Solution:
	def deleteNode(self, node):
		node.val = node.next.val
		node.next = node.next.next

# 238 Product of Array Except Self
# Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].
class Solution:
	def productExceptSelf(self, nums):
		output = [1]*len(nums)
		left = 1
		for x in range(len(nums)-1):
			left *= nums[x]
			output[x+1] *= left
		right = 1
		for x in range(len(nums)-1, 0, -1):
			right *= nums[x]
			output[x-1] *= right
		return output

# 239 Sliding Window Maximum
# Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right.
# You can only see the k numbers in the window. Each time the sliding window moves right by one position.
class Solution:
	def maxSlidingWindow(self, nums, k):
		dq, ans = collections.deque(), []
		for i in range(len(nums)):
			while dq and nums[dq[-1]] <= nums[i]:
				dq.pop()
			dq.append(i)
			if dq[0] == i-k:
				dq.popleft()
			if i >= k-1:
				ans.append(nums[dq[0]])
		return ans

# 240 Search a 2D Matrix II
# Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
# Integers in each row are sorted in ascending from left to right.
# Integers in each column are sorted in ascending from top to bottom.
[
  [ 1,  4,  7, 11, 15],
  [ 2,  5,  8, 12, 19],
  [ 3,  6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
class Solution:
	def searchMatrix(self, matrix, target):
		y = len(matrix[0])-1
		for x in range(len(matrix)):
			while y and matrix[x][y] > target:
				y -= 1
			if matrix[x][y] == target:
				return True
		return False

# 241 Different Ways to Add Parentheses
# Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators.
# The valid operators are +, - and *.
class Solution:
	def diffWaysToCompute(self, input):
		return [a+b if c == '+' else a-b if c == '-' else a*b
			for i, c in enumerate(input) if c in '+-*'
			for a in self.diffWaysToCompute(input[:i])
			for b in self.diffWaysToCompute(input[i+1:])] or [int(input)]

# 242 Valid Anagram
# Given two strings s and t, write a function to determine if t is an anagram of s.
class Solution:
	def isAnagram(self, s, t):
		return sorted(s) == sorted(t)

# 243 Shortest Word Distance
# Given a list of words and two words word1 and word2, return the shortest distance between these two words in the list.
class Solution:
	def shortestDistance(self, words, word1, word2):
		dist = float("inf")
		i, index1, index2 = 0, None, None
		while i < len(words):
			if words[i] == word1:
				index1 = i
			elif words[i] == word2:
				index2 = i
			if index1 and index2:
				dist = min(dist, abs(index1 - index2))
			i += 1
		return dist

# 244 Shortest Word Distance II
# your method will be called repeatedly many times with different parameters. How would you optimize it?
class WordDistance:
	def __init__(self, words):
		self.wordIndex = collections.defaultdict(list)
		for i in xrange(len(words)):
			self.wordIndex[words[i]].append(i)
	def shortest(self, word1, word2):
		indexes1 = self.wordIndex[word1]
		indexes2 = self.wordIndex[word2]
		i, j, dist = 0, 0, float("inf")
		while i < len(indexes1) and j < len(indexes2):
			dist = min(dist, abs(indexes1[i]-indexes2[j]))
			if indexes1[i] < indexes2[j]: i += 1
			else: j += 1
		return dist

# 245 Shortest Word Distance III
# word1 and word2 may be the same and they represent two individual words in the list.
class Solution:
	def shortestWordDistance(self, words, word1, word2):
		dist = float("inf")
		i, index1, index2 = 0, None, None
		while i < len(words):
			if words[i] == word1:
				if index1:
					dist = min(dist, abs(index1-i))
				index1 = i
			elif words[i] == word2:
				index2 = i
			if index1 and index2:
				dist = min(dist, abs(index1-index2))
			i += 1
		return dist

# 246 Strobogrammatic Number
# A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
# Write a function to determine if a number is strobogrammatic. The number is represented as a string.
class Solution:
	lookup = {'0':'0', '1':'1', '6':'9', '8':'8', '9':'6'}
	def isStrobogrammatic(self, num):
		n = len(num)
		for i in xrange((n+1)/2):
			if num[n-1-i] not in self.lookup or \
			   num[i] != self.lookup[num[n-1-i]]:
				return False
			i += 1
		return True

# 247 Strobogrammatic Number II
# Find all strobogrammatic numbers that are of length = n.
class Solution:
	lookup = {'0':'0', '1':'1', '6':'9', '8':'8', '9':'6'}
	def findStrobogrammatic(self, n):
		return self.findStrobogrammaticRecu(n, n)
	def findStrobogrammaticRecu(self, n, k):
		if k == 0:
			return ['']
		elif k == 1:
			return ['0', '1', '8']
		result = []
		for num in self.findStrobogrammaticRecu(n, k-2):
			for key, val in self.lookup.iteritems():
				if n != k or key != '0':
					result.append(key+num+val)
		return result

# 248 Strobogrammatic Number III
# Write a function to count the total strobogrammatic numbers that exist in the range of low <= num <= high.
class Solution:
	def strobogrammaticInRange(self, low, high):
		a = self.below(high)
		b = self.below(low,include=False)
		return a-b if a>b else 0
	def below(self,n,include=True):
		res = 0
		for i in range(1,len(n)):
			res += self.number(i)
		l = self.strobogrammatic(len(n))
		if include:
			l = [num for num in l if (len(num)==1 or num[0]!='0') and num<=n]
		else:
			l = [num for num in l if (len(num)==1 or num[0]!='0') and num<n]
		return res+len(l)
	def strobogrammatic(self,l):
		res = []
		if l == 1:
			return ['0','1','8']
		if l == 2:
			return ['00','11','69','96','88']
		for s in self.strobogrammatic(l-2):
			res.append('0'+s+'0')
			res.append('1'+s+'1')
			res.append('6'+s+'9')
			res.append('8'+s+'8')
			res.append('9'+s+'6')
		return res
	def number(self,l):
		'''
		If l is an even number, the first digit has four choices (1,6,8,9). digits 
		at other position have five choices(0,1,6,8,9)
		'''
		'''
		If l is an odd number, the first digit has four choices (1,6,8,9) and digit 
		at the middle has 3 choices (0,1,8),other digits have 5 choices.
		digit at other position could be 0,1,6,8,9
		'''
		if l==0:
			return 0
		if l%2==0:
			return 4*(5**(l/2-1))
		elif l==1:
			return 3
		else:
			return 3*(5**(l/2-1))*4

# 249 Group Shifted Strings
# Given a string, we can “shift” each of its letter to its successive letter, for example: “abc” -> “bcd”. We can keep “shifting” which forms the sequence:
# "abc" -> "bcd" -> ... -> "xyz"
# Given a list of strings which contains only lowercase alphabets, group all strings that belong to the same shifting sequence.
class Solution:
	def groupStrings(self, strings):
		dic = {}
		for s in strings:
			# "abc"->(0,1,2), "az"->(0,25), etc 
			tmp = tuple(map(lambda x:(ord(x)-ord(s[0]))%26, s))
			dic[tmp] = dic.get(tmp, [])+[s]
		return [sorted(x) for x in dic.values()]

# 250 Count Univalue Subtrees
# Given a binary tree, count the number of uni-value subtrees.
# A Uni-value subtree means all nodes of the subtree have the same value.
class Solution:
	def countUnivalSubtrees(self, root):
		[is_uni, count] = self.isUnivalSubtrees(root, 0)
		return count
	def isUnivalSubtrees(self, root, count):
		if not root: return [True, count]
		[left, count] = self.isUnivalSubtrees(root.left, count)
		[right, count] = self.isUnivalSubtrees(root.right, count)
		if self.isSame(root, root.left, left) and \
		   self.isSame(root, root.right, right):
				count += 1; return [True, count]
		return [False, count]
	def isSame(self, root, child, is_uni):
		return not child or (is_uni and root.val == child.val)

# 251 Flatten 2D Vector
# Implement an iterator to flatten a 2d vector.
class Vector2D:
	def __init__(self, vec2d):
		self.q = vec2d
		self.x, self.y = -1, 0
		self._move_down()
	def _move_down(self):
		self.x += 1
		while self.x < len(self.q) and len(self.q[self.x]) == 0:
			self.x += 1
	def next(self):
		ret = self.q[self.x][self.y]
		if self.y+1 == len(self.q[self.x]):
			self.y = 0; self._move_down()
		else:
			self.y += 1
		return ret
	def hasNext(self):
		return self.x < len(self.q) and self.y < len(self.q[self.x])

# 252 Meeting Rooms 
# Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.
class Solution:
	def canAttendMeetings(self, intervals):
		intervals.sort(key=lambda x: x.start)
		for i in xrange(1, len(intervals)):
			if intervals[i].start < intervals[i-1].end:
				return False
		return True

# 253 Meeting Rooms II
# Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.
class Solution:
	def minMeetingRooms(self, intervals):
		intervals.sort(key=lambda x:x.start)
		heap = []
		for i in intervals:
			if heap and i.start >= heap[0]:
				heapq.heapreplace(heap, i.end)
			else:
				heapq.heappush(heap, i.end)
		return len(heap)

# 254 Factor Combinations
# Numbers can be regarded as product of its factors. For example, 8 = 2 x 2 x 2 = 2 x 4.
# Write a function that takes an integer n and return all possible combinations of its factors.
class Solution:
	def getFactors(self, n):
		if n <= 1: return []
		res, i = [], 2
		while i*i <= n:
			if n%i == 0:
				q = n/i
				res.append([i, q])
				subres = self.getFactors(q)
				for r in subres:
					if r[0] >= i:
						res.append([i]+r)
			i += 1
		return res

# 255 Verify Preorder Sequence in Binary Search Tree
# Given an array of numbers, verify whether it is the correct preorder traversal sequence of a binary search tree.
# You may assume each number in the sequence is unique.
class Solution:
	def verifyPreorder(self, preorder):
		inorder, stack = [], []
		for p in preorder:
			if inorder and p < inorder[-1]:
				return False
			while stack and p > stack[-1]:
				inorder.append(stack.pop())
			stack.append(p)
		return True

# 256 Paint House
# There are a row of n houses, each house can be painted with one of the three colors: red, blue or green.
# The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.
# The cost of painting each house with a certain color is represented by a n x 3 cost matrix. Find the minimum cost to paint all houses.
class Solution:
	def minCost(self, costs):
		size = len(costs)
		if size == 0: return 0
		pre = costs[0][:]
		now = [0]*3
		for i in xrange(size-1):
			now[0] = min(pre[1], pre[2]) + costs[i+1][0]
			now[1] = min(pre[0], pre[2]) + costs[i+1][1]
			now[2] = min(pre[0], pre[1]) + costs[i+1][2]
			pre[:] = now[:]
		return min(pre)

# 257 Binary Tree Paths
# Given a binary tree, return all root-to-leaf paths.
class Solution:
	def binaryTreePaths(self, root):
		if not root: return []
		return [str(root.val) + '->' + path
				for kid in (root.left, root.right) if kid
				for path in self.binaryTreePaths(kid)] or [str(root.val)]

# 258 Add Digits
# Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
class Solution:
	def addDigits(self, num): # 观察法
		if num == 0: return 0
		return (num-1)%9+1

# 259 3Sum Smaller 
# Given an array of n integers nums and a target, find the number of index triplets i, j, k with 0 <= i < j < k < n that satisfy the condition nums[i] + nums[j] + nums[k] < target.
class Solution:
	def threeSumSmaller(self, a, target):
		a.sort()
		ans, n = 0, len(a)
		for i in xrange(n-2):
			j, k = i+1, n-1
			while j < k:
				if a[i]+a[j]+a[k] < target:
					ans += k-j
					j += 1
				else:
					k -= 1
		return ans

# 260 Single Number III
# Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice.
# Find the two elements that appear only once.
class Solution:
	def singleNumber(self, nums):
		xor = reduce(lambda x,y:x^y, nums)
		lowbit = xor & -xor
		a = b = 0
		for num in nums:
			if num & lowbit:
				a ^= num
			else:
				b ^= num
		return [a, b]

# 261 Graph Valid Tree
# Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), 
# write a function to check whether these edges make up a valid tree.
# For example:
# Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return true.
# Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], return false.
def validTree(self, n, edges):
	dic = {i: set() for i in xrange(n)}
	for i, j in edges:
		dic[i].add(j)
		dic[j].add(i)
	stack, visited = [dic.keys()[0]], set()
	while stack:
		node = stack.pop()
		if node in visited:
			return False
		visited.add(node)
		for neighbour in dic[node]:
			stack.append(neighbour)
			dic[neighbour].remove(node)
		dic.pop(node)
	return not dic

# 263 Ugly Number
# Write a program to check whether a given number is an ugly number.
# Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. For example, 6, 8 are ugly while 14 is not ugly since it includes another prime factor 7.
# Note that 1 is typically treated as an ugly number.
class Solution:
	def isUgly(self, num):
		if num <= 0:
			return False
		for x in [2, 3, 5]:
			while num % x == 0:
				num /= x
		return num == 1

# 264 Ugly Number II
# Write a program to find the n-th ugly number.
# Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. For example, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 is the sequence of the first 10 ugly numbers.
# Note that 1 is typically treated as an ugly number.
class Solution:
	def nthUglyNumber(self, n):
		q = [1]
		i2 = i3 = i5 = 0
		while len(q) < n:
			m2, m3, m5 = q[i2]*2, q[i3]*3, q[i5]*5
			m = min(m2, m3, m5)
			if m == m2:
				i2 += 1
			if m == m3:
				i3 += 1
			if m == m5:
				i5 += 1
			q += m,
		return q[-1]

# 265 Paint House II 
# There are a row of n houses, each house can be painted with one of the k colors. The cost of painting each house with a certain color is different. 
# You have to paint all the houses such that no two adjacent houses have the same color.
# The cost of painting each house with a certain color is represented by a n x k cost matrix. Find the minimum cost to paint all houses.
class Solution:
	def minCostII(self, costs):
		if not costs: return 0
		r, c = len(costs), len(costs[0])
		cur = costs[0]
		for i in xrange(1, r):
			pre = cur[:]
			for j in xrange(c):
				cur[j] = costs[i][j] + min(pre[:j]+pre[j+1:])
		return min(cur)

# 266 Palindrome Permutation
# Given a string, determine if a permutation of the string could form a palindrome.
class Solution:
	def canPermutePalindrome(self, s):
		return sum(v%2 for v in collections.Counter(s).values()) < 2

# 267 Palindrome Permutation II
# Given a string s, return all the palindromic permutations (without duplicates) of it. Return an empty list if no palindromic permutation could be form.
class Solution:
	def generatePalindromes(self, s):
		d = collections.Counter(s)
		m = tuple(k for k, v in d.iteritems() if v%2)
		p = ''.join(k*(v/2) for k, v in d.iteritems())
		return [''.join(i+m+i[::-1]) for i in set(itertools.permutations(p))] if len(m) < 2 else []

# 268 Missing Number
# Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.
class Solution:
	def missingNumber(self, nums):
		n = len(nums)
		return n*(n+1)/2 - sum(nums)

# 269 Alien Dictionary
# There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. 
# You receive a list of words from the dictionary, wherewords are sorted lexicographically by the rules of this new language. Derive the order of letters in this language.
class Solution:
	def alienOrder(self, words):
		less = []
		for pair in zip(words[:-1], words[1:]):
			for a, b in zip(*pair):
				if a != b:
					less += a+b,
					break
		chars, order = set(''.join(words)), []
		while less:
			free = chars-set(zip(*less)[1])
			if not free: return ''
			order += free
			less = filter(free.isdisjoint, less)
			chars -= free
		return ''.join(order+list(chars))

# 270 Closest Binary Search Tree Value
# Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.
class Solution:
	def closestValue(self, root, target):
		a = root.val
		kid = root.left if target < a else root.right
		if not kid: return a
		b = self.closestValue(kid, target)
		return min((b, a), key=lambda x:abs(target-x))

# 271 Encode and Decode Strings
# Design an algorithm to encode a list of strings to a string. 
# The encoded string is then sent over the network and is decoded back to the original list of strings.
class Codec:
	def encode(self, strs):
		return ''.join('%d:'%len(s)+s for s in strs)
	def decode(self, s):
		strs, i = [], 0
		while i < len(s):
			j = s.find(':', i)
			i = j+1+int(s[i:j])
			strs.append(s[j+1:i])
		return strs

# 272 Closest Binary Search Tree Value II
# Given a non-empty binary search tree and a target value, find k values in the BST that are closest to the target.
class Solution:
	def closestKValues(self, root, target, k): # 中序遍历 + 堆
		values, stack = [], []
		while stack or root:
			if root:
				stack.append(root)
				root = root.left
			else:
				root = stack.pop()
				heapq.heappush(values, (abs(root.val-target),root.val))
				root = root.right
		return [heapq.heappop(values)[1] for _ in range(k)]

# 273 Integer to English Words
# Convert a non-negative integer to its english words representation. Given input is guaranteed to be less than 2^31 - 1.
class Solution:
	def numberToWords(self, num):
		lv1 = "Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen".split()
		lv2 = "Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety".split()
		lv3 = "Hundred"
		lv4 = "Thousand Million Billion".split()
		words, digits = [], 0
		while num:
			token, num = num%1000, num/1000
			word = ''
			if token > 99:
				word += lv1[token/100]+' '+lv3+' '
				token %= 100
			if token > 19:
				word += lv2[token/10-2]+' '
				token %= 10
			if token > 0:
				word += lv1[token]+' '
			word = word.strip()
			if word:
				word += ' '+lv4[digits-1] if digits else ''
				words += word,
			digits += 1
		return ' '.join(words[::-1]) or 'Zero'

# 274 H-Index
# Given an array of citations (each citation is a non-negative integer) of a researcher, write a function to compute the researcher's h-index.
# According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each."
class Solution:
	def hIndex(self, citations):
		return sum(i < c for i, c in enumerate(sorted(citations, reverse=True)))

# 275 H-Index II
# Follow up for H-Index: What if the citations array is sorted in ascending order? Could you optimize your algorithm?
class Solution:
	def hIndex(self, citations):
		N = len(citations)
		low, high = 0, N-1
		while low <= high:
			mid = (low+high)/2
			if N-mid > citations[mid]:
				low = mid+1
			else:
				high = mid-1
		return N-low

# 276 Paint Fence
# There is a fence with n posts, each post can be painted with one of the k colors.
# You have to paint all the posts such that no more than two adjacent fence posts have the same color.
# Return the total number of ways you can paint the fence.
class Solution:
	def numWays(self, n, k):
		if n == 0:
			return 0
		if n == 1:
			return k
		same, dif = k, k*(k-1)
		for i in range(3, n+1):
			same, dif = dif, (same+dif)*(k-1)
		return same+dif

# 277 Find the Celebrity
# Suppose you are at a party with n people (labeled from 0 to n - 1) and among them, there may exist one celebrity. 
# The definition of a celebrity is that all the other n - 1 people know him/her but he/she does not know any of them.
# You are given a helper function bool knows(a, b) which tells you whether A knows B. Implement a function int findCelebrity(n), your function should minimize the number of calls to knows.
class Solution:
	def findCelebrity(self, n):
		x = 0
		for i in xrange(n):
			if knows(x,i):
				x = i
		if any(knows(x,i) for i in xrange(x)):
			return -1
		if any(not knows(i,x) for i in xrange(n)):
			return -1
		return x

# 278 First Bad Version
# Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.
# You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.
class Solution:
	def firstBadVersion(self, n):
		left, right = 1, n
		while left <= right:
			mid = (left+right)/2
			if isBadVersion(mid):
				right = mid-1
			else:
				left = mid+1
		return left

# 279 Perfect Squares
# Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.
# For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, return 2 because 13 = 4 + 9.
class Solution:
	_dp = [0]
	def numSquares(self, n):
		dp = self._dp
		while len(dp) <= n:
			dp += min(dp[-i*i] for i in range(1, int(len(dp)**0.5+1))) + 1,
		return dp[n]

# 280 Wiggle Sort
# Given an unsorted array nums, reorder it in-place such that nums[0] <= nums[1] >= nums[2] <= nums[3]....
class Solution:
	def wiggleSort(self, a):
		for i in xrange(0, len(a), 2):
			s = sorted(a[i:i+3])
			a[i:i+3] = s[:1]+s[:0:-1]

# 281 Zigzag Iterator
# Given two 1d vectors, implement an iterator to return their elements alternately.
class Solution:
	def __init__(self, v1, v2):
		self.v1 = v1
		self.v2 = v2
	def next(self):
		if self.v1 or self.v2:
			if self.v1:
				res = self.v1.pop(0)
				if self.v2:
					self.v1, self.v2 = self.v2, self.v1
			else:
				res = self.v2.pop(0)
				self.v1, self.v2 = self.v2, self.v1
			return res
		return []
	def hasNext(self):
		return len(self.v1) + len(self.v2) > 0

# 282 Expression Add Operators
# Given a string that contains only digits 0-9 and a target value, return all possibilities to add binary operators (not unary) +, -, or * between the digits so they evaluate to the target value.
class Solution:
	def addOperators(self, num, target):
		def isLeadingZeros(num):
			return num.startswith('00') or int(num) and num.startswith('0')
		def solve(num, target, mulExpr = '', mulVal = 1):
			ans = []
			if isLeadingZeros(num):
				pass
			elif int(num) * mulVal == target:
				ans += num+mulExpr,
			for x in range(len(num)-1):
				lnum, rnum = num[:x+1], num[x+1:]
				if isLeadingZeros(rnum):
					continue
				right, rightVal = rnum+mulExpr, int(rnum)*mulVal
				for left in solve(lnum, target-rightVal): #op = '+'
					ans += left+'+'+right,
				for left in solve(lnum, target+rightVal): #op = '-'
					ans += left+'-'+right,
				for left in solve(lnum, target, '*'+right, rightVal): #op = '*'
					ans += left,
			return ans
		if not num:
			return []
		return solve(num, target)

# 283 Move Zeroes
# Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.
class Solution:
	def moveZeroes(self, nums):
		y = 0
		for x in range(len(nums)):
			if nums[x]:
				nums[x], nums[y] = nums[y], nums[x]
				y += 1

# 284 Peeking Iterator
# Given an Iterator class interface with methods: next() and hasNext(), design and implement a PeekingIterator that support the peek() operation 
# -- it essentially peek() at the element that will be returned by the next call to next().
class Iterator:
	def __init__(self, nums):
		"""
		Initializes an iterator object to the beginning of a list.
		:type nums: List[int]
		"""
	def hasNext(self):
		"""
		Returns true if the iteration has more elements.
		:rtype: bool
		"""
	def next(self):
		"""
		Returns the next element in the iteration.
		:rtype: int
		"""
class PeekingIterator:
	def __init__(self, iterator):
		self.iter = iterator
		self.peekFlag = False
		self.nextElement = None
	def peek(self):
		if not self.peekFlag:
			self.nextElement = self.iter.next()
			self.peekFlag = True
		return self.nextElement
	def next(self):
		if not self.peekFlag:
			return self.iter.next()
		else:
			nextElement = self.nextElement
			self.peekFlag = False
			self.nextElement = None
			return nextElement
	def hasNext(self):
		return self.peekFlag or self.iter.hasNext()

# 285 Inorder Successor in BST leetcode
# Given a binary search tree and a node in it, find the in-order successor of that node in the BST.
class Solution:
	def inorderSuccessor(self, root, p):
		succ = None
		while root:
			if p.val < root.val:
				succ = root
				root = root.left
			else:
				root = root.right
		return succ

# 286 Walls and Gates
# You are given a m x n 2D grid initialized with these three possible values.
# -1 - A wall or an obstacle.
# 0 - A gate.
# INF - Infinity means an empty room. We use the value 2**31 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than2147483647.
# Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.
class Solution:
	def wallsAndGates(self, rooms):
		q = [(i, j) for i, row in enumerate(rooms) for j, r in enumerate(row) if not r]
		for i, j in q:
			for x, y in (i+1,j), (i-1,j), (i,j+1), (i,j-1):
				if 0 <= x < len(rooms) and 0 <= y < len(rooms[0]) and rooms[x][y] > 2**30:
					rooms[x][y] = rooms[i][j]+1
					q += (x,y),

# 287 Find the Duplicate Number
# Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist.
# Assume that there is only one duplicate number, find the duplicate one.
class Solution:
	def findDuplicate(self, nums):
		low, high = 1, len(nums)-1
		while low <= high:
			mid = (low+high)/2
			cnt = sum(x <= mid for x in nums)
			if cnt > mid:
				high = mid-1
			else:
				low = mid+1
		return low

# 288 Unique Word Abbreviation
# An abbreviation of a word follows the form <first letter><number><last letter>.
# Assume you have a dictionary and given a word, find whether its abbreviation is unique in the dictionary.
class ValidWordAbbr:
	def __init__(self, dictionary):
		self.dt = collections.defaultdict(set)
		for d in dictionary:
			abbr = d[0]+str(len(d))+d[-1]
			self.dt[abbr].add(d)
	def isUnique(self, word):
		abbr = word[0]+str(len(word))+word[-1]
		return abbr not in self.dt or self.dt[abbr] == set([word])

# 289 Game of Life
# According to the Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."
# Given a board with m by n cells, each cell has an initial state live (1) or dead (0). 
# Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):
# Any live cell with fewer than two live neighbors dies, as if caused by under-population.
# Any live cell with two or three live neighbors lives on to the next generation.
# Any live cell with more than three live neighbors dies, as if by over-population..
# Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
# Write a function to compute the next state (after one update) of the board given its current state.
class Solution:
	def gameOfLife(self, board):
		dx = (1, 1, 1, 0, 0, -1, -1, -1)
		dy = (1, 0, -1, 1, -1, 1, 0, -1)
		for x in range(len(board)):
			for y in range(len(board[0])):
				lives = 0
				for z in range(8):
					nx, ny = x+dx[z], y+dy[z]
					lives += self.getCellStatus(board,nx,ny)
				if lives+board[x][y] == 3 or lives == 3:
					board[x][y] |= 2
		for x in range(len(board)):
			for y in range(len(board[0])):
				board[x][y] >>= 1
	def getCellStatus(self, board, x, y):
		if x<0 or y<0 or x>=len(board) or y>=len(board[0]):
			return 0
		return board[x][y]&1

# 290 Word Pattern
# Given a pattern and a string str, find if str follows the same pattern.
# Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in str.
class Solution:
	def wordPattern(self, pattern, str):
		s = pattern
		t = str.split()
		return len(set(zip(s,t))) == len(set(s)) == len(set(t))

# 291 Word Pattern II 
# Given a pattern and a string str, find if str follows the same pattern.
# Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty substring in str.
class Solution:
	def wordPatternMatch(self, pattern, str):
		return self.dfs(pattern, str, {})
	def dfs(self, pattern, str, dict):
		if len(pattern) == 0 and len(str) > 0: return False
		if len(pattern) == len(str) == 0: return True
		for end in range(1, len(str)-len(pattern)+2): # +2 because it is the "end of an end"
			if pattern[0] not in dict and str[:end] not in dict.values():
				dict[pattern[0]] = str[:end]
				if self.dfs(pattern[1:], str[end:], dict):
					return True
				del dict[pattern[0]]
			elif pattern[0] in dict and dict[pattern[0]] == str[:end]:
				if self.dfs(pattern[1:], str[end:], dict):
					return True
		return False

# 292 Nim Game
# You are playing the following Nim Game with your friend: There is a heap of stones on the table, each time one of you take turns to remove 1 to 3 stones. 
# The one who removes the last stone will be the winner. You will take the first turn to remove the stones.
# Both of you are very clever and have optimal strategies for the game. Write a function to determine whether you can win the game given the number of stones in the heap.
class Solution:
	def canWinNim(self, n):
		return n%4 > 0

# 293 Flip Game
# You are playing the following Flip Game with your friend: Given a string that contains only these two characters: + and -, you and your friend take turns to flip two consecutive "++" into "--".
# Write a function to compute all possible states of the string after one valid move.
class Solution:
	def generatePossibleNextMoves(self, s):
		return [s[:i]+"--"+s[i+2:] for i in xrange(len(s)-1) if s[i:i+2]=='++']

# 294 Flip Game II
# Write a function to determine if the starting player can guarantee a win.
class Solution:
	def canWin(self, s):
		memo = {}
		def can(s):
			if s not in memo:
				memo[s] = any(s[i:i+2] == '++' and not can(s[:i]+'--'+s[i+2:]) for i in range(len(s)))
			return memo[s]
		return can(s)

# 295 Find Median from Data Stream
# Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle value.
import bisect
class MedianFinder:
	def __init__(self):
		self.nums = []
	def addNum(self, num):
		bisect.insort(self.nums, num)
	def findMedian(self):
		nums = self.nums
		if len(nums)%2 == 0:
			return (nums[len(nums)/2]+nums[len(nums)/2-1])/2.0
		else:
			return nums[len(nums)/2]

# 296 Best Meeting Point
# A group of two or more people wants to meet and minimize the total travel distance. You are given a 2D grid of values 0 or 1, where each 1 marks the home of someone in the group. 
# The distance is calculated using Manhattan Distance, where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.
class Solution:
	def minTotalDistance(self, grid):
		x = sorted([i for i, row in enumerate(grid) for v in row if v == 1])
		y = sorted([j for row in grid for j, v in enumerate(row) if v == 1])
		return sum([abs(x[len(x)/2]-i)+abs(y[len(y)/2]-j) for i, row in enumerate(grid) for j, v in enumerate(row) if v == 1])

# 297 Serialize and Deserialize Binary Tree
class Codec:
	def serialize(self, root):
		def doit(node):
			if node:
				vals.append(str(node.val))
				doit(node.left)
				doit(node.right)
			else:
				vals.append('#')
		vals = []
		doit(root)
		return ' '.join(vals)
	def deserialize(self, data):
		def doit():
			val = next(vals)
			if val == '#': return None
			node = TreeNode(int(val))
			node.left = doit()
			node.right = doit()
			return node
		vals = iter(data.split())
		return doit()

# 298 Binary Tree Longest Consecutive Sequence
# Given a binary tree, find the length of the longest consecutive sequence path.
# The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The longest consecutive path need to be from parent to child (cannot be the reverse).
class Solution:
	def longestConsecutive(self, root):
		return self.helper(root)[0]
	def helper(self, root):
		if not root: return 0, 0
		g_l, l_l = self.helper(root.left)
		g_r, l_r = self.helper(root.right)
		local = 1
		if root.left and root.val == root.left.val-1:
			local = l_l+1
		if root.right and root.val == root.right.val-1:
			local = max(local,l_r+1)
		return max(g_l,g_r,local), local

# 299 Bulls and Cows
# You are playing the following Bulls and Cows game with your friend: You write a 4-digit secret number and ask your friend to guess it. 
# Each time your friend guesses a number, you give a hint. The hint tells your friend how many digits are in the correct positions (called "bulls") and how many digits are in the wrong positions (called "cows"). 
# Your friend will use those hints to find out the secret number.
class Solution:
	def getHint(self, secret, guess):
		bull = sum(map(operator.eq, secret, guess))
		sa = collections.Counter(secret)
		sb = collections.Counter(guess)
		cow = sum((sa&sb).values())-bull
		return str(bull)+'A'+str(cow)+'B'

# 300 Longest Increasing Subsequence
# Given an unsorted array of integers, find the length of longest increasing subsequence.
# For example,
# Given [10, 9, 2, 5, 3, 7, 101, 18],
# The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. Note that there may be more than one LIS combination, it is only necessary for you to return the length.
# Your algorithm should run in O(n2) complexity.
class Solution:
	def lengthOfLIS(self, nums):
		size = len(nums)
		dp = [1] * size
		for x in range(size):
			for y in range(x):
				if nums[x] > nums[y]:
					dp[x] = max(dp[x],dp[y]+1)
		return max(dp) if dp else 0

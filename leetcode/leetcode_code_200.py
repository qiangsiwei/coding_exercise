# -*- coding: utf-8 -*- 

# 101 Symmetric Tree
# Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
# For example, this binary tree is symmetric:
#	  1
#	 / \
#   2   2
#  / \ / \
# 3  4 4  3
# But the following is not:
#	 1
#	/ \
#  2   2
#	\   \
#	3	3
# Note: Bonus points if you could solve it both recursively and iteratively.
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def isSymmetric(self, root):
		if root == None: return True
		return self.checkSym(root.left, root.right)
	def checkSym(self, left, right):
		if left == right == None: return True
		if not (left and right): return False
		if left.val != right.val: return False
		return self.checkSym(left.left, right.right) and self.checkSym(left.right, right.left)

# 102 Binary Tree Level Order Traversal
# Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).
# For example:
# Given binary tree {3,9,20,#,#,15,7},
#	 3
#	/ \
#  9  20
#	 /  \
#	15   7
# return its level order traversal as:
# [
#   [3],
#   [9,20],
#   [15,7]
# ]
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def preOrder(self, root, level):
		if level in Solution.L:
			Solution.L[level].append(root.val)
		else:
			Solution.L[level] = [root.val]
		if root.left:
			self.preOrder(root.left, level+1)
		if root.right:
			self.preOrder(root.right, level+1)
		return
	def levelOrder(self, root):
		res = []
		if root == None: return res
		Solution.L = {}
		self.preOrder(root, 0)
		for i in sorted(Solution.L.keys()):
			res.append(Solution.L[i])
		return res

# 103 Binary Tree Zigzag Level Order Traversal
# Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).
# For example:
# Given binary tree {3,9,20,#,#,15,7},
#	 3
#	/ \
#  9  20
#	 /  \
#	15   7
# return its zigzag level order traversal as:
# [
#   [3],
#   [20,9],
#   [15,7]
# ]
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def preOrder(self, root, level):
		if level not in Solution.L:
			Solution.L[level] = [root.val]
		else:
			Solution.L[level].append(root.val)
		if root.left:
			self.preOrder(root.left, level+1)
		if root.right:
			self.preOrder(root.right, level+1)
	def zigzagLevelOrder(self, root):
		res = []
		if root == None: return res
		Solution.L = {}
		self.preOrder(root, 0)
		for i in sorted(Solution.L.keys()):
			if i%2 == 0:
				res.append(Solution.L[i])
			else:
				res.append(Solution.L[i][::-1])
		return res

# 104 Maximum Depth of Binary Tree
# Given a binary tree, find its maximum depth.
# The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def maxDepth(self, root):
		if root is None:
			return 0
		else:	
			return max(self.maxDepth(root.left), self.maxDepth(root.right))+1

# 105 Construct Binary Tree
# Given preorder and inorder traversal of a tree, construct the binary tree.
class Solution:
	def buildTree(self, preorder, inorder):
		if inorder:
			ind = inorder.index(preorder.pop(0))
			root = TreeNode(inorder[ind])
			root.left = self.buildTree(preorder, inorder[0:ind])
			root.right = self.buildTree(preorder, inorder[ind+1:])
			return root

# 106 Construct Binary Tree
# Given inorder and postorder traversal of a tree, construct the binary tree.
class Solution:
	def buildTree(self, inorder, postorder):
		if not inorder: return None
		ind = inorder.index(postorder.pop())
		root = TreeNode(inorder[ind])
		root.right = self.buildTree(inorder[ind+1:len(inorder)], postorder)
		root.left = self.buildTree(inorder[:ind], postorder)
		return root

# 107 Binary Tree Level Order Traversal II
# Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).
# For example:
# Given binary tree {3,9,20,#,#,15,7},
#	 3
#	/ \
#  9  20
#	 /  \
#	15   7
# return its bottom-up level order traversal as:
# [
#   [15,7],
#   [9,20],
#   [3]
# ]
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def preOrder(self, root, level):
		if level not in Solution.L:
			Solution.L[level] = [root.val]
		else:
			Solution.L[level].append(root.val)
		if root.left:
			self.preOrder(root.left, level+1)
		if root.right:
			self.preOrder(root.right, level+1)
	def levelOrderBottom(self, root):
		res = []
		if root == None: return res
		Solution.L = {}
		self.preOrder(root, 0)
		for i in sorted(Solution.L.keys(), reverse=True):
			res.append(Solution.L[i])
		return res

# 108 Convert Sorted Array to Binary Search Tree
# Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def sortedArrayToBST(self, num):
		return self.createBST(num, 0, len(num)-1)
	def createBST(self, num, start, end):
		if start > end: return None
		mid = (start+end)/2
		root = TreeNode(num[mid])
		root.left = self.createBST(num, start, mid-1)
		root.right = self.createBST(num, mid+1, end)
		return root

# 109 Convert Sorted List to Binary Search Tree
# Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class ListNode:
	def __init__(self, x):
		self.val = x
		self.next = None
class Solution:
	def sortedListToBST(self, head):
		num = []; curr = head
		while curr != None: 
			num.append(curr.val)
			curr = curr.next
		return self.createBST(num, 0, len(num)-1)
	def createBST(self, num, start, end):
		if start > end: return None
		mid = (start+end)/2
		root = TreeNode(num[mid])
		root.left = self.createBST(num, start, mid-1)
		root.right = self.createBST(num, mid+1, end)
		return root

# 110 Balanced Binary Tree
# Given a binary tree, determine if it is height-balanced.
# For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def isBalanced(self, root):
		return self.check(root)[1]
	def check(self, root):
		if root == None: return (0, True)
		LH, LisB = self.check(root.left)
		RH, RisB = self.check(root.right)
		return (max(LH, RH)+1, LisB and RisB and abs(LH-RH)<=1)

# 111 Minimum Depth of Binary Tree
# Given a binary tree, find its minimum depth.
# The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
def __init__(self, x):
	self.val = x
	self.left = None
	self.right = None
class Solution:
	def minDepth(self, root):
		if root == None: 
			return 0
		if root.left == None and root.right == None:
			return 1
		if root.left == None:
			return self.minDepth(root.right)+1
		if root.right == None:
			return self.minDepth(root.left)+1
		return min(self.minDepth(root.left), self.minDepth(root.right))+1

# 112 Path Sum
# Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.
# For example:
# Given the below binary tree and sum = 22,
#			   5
#			  / \
#			 4   8
#			/   / \
#		   11  13  4
#		  /  \	 \
#		 7	 2	  1
# return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def hasPathSum(self, root, sum):
		if root == None: return False
		if root.val == sum and root.left == None and root.right == None: return True
		if root.left != None and self.hasPathSum(root.left, sum-root.val): return True
		if root.right != None and self.hasPathSum(root.right, sum-root.val): return True
		return False

# 113 Path Sum II
# Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.
# For example:
# Given the below binary tree and sum = 22,
#			   5
#			  / \
#			 4   8
#			/   / \
#		   11  13  4
#		  /  \	/ \
#		 7	2  5   1
# return
# [
#	[5,4,11,2],
#	[5,8,4,5]
# ]
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def pathSum(self, root, Sum):
		if root == None: return []
		Solution.res = []
		Solution.Sum = Sum
		self.getPath(root, [root.val], root.val)
		return Solution.res
	def getPath(self, root, valList, currSum):
		if root.left == None and root.right == None:
			if currSum == Solution.Sum: 
				Solution.res.append(valList); return
		if root.left:
			self.getPath(root.left, valList+[root.left.val], currSum+root.left.val)
		if root.right:
			self.getPath(root.right, valList+[root.right.val], currSum+root.right.val)

# 114 Flatten Binary Tree to Linked List
# Given a binary tree, flatten it to a linked list in-place.
# For example,
# Given
#		  1
#		 / \
#		2   5
#	   / \   \
#	  3   4   6
# The flattened tree should look like:
#	1
#	 \
#	  2
#	   \
#		3
#		 \
#		  4
#		   \
#			5
#			 \
#			  6
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def flatten(self, root):
		# eliminate each level's root's left child
		while root:
			if root.left:
				p = root.left
				while p.right: p = p.right
				p.right = root.right
				root.right = root.left
				root.left = None
			root = root.right

# 115 Distinct Subsequences
# Given a string S and a string T, count the number of distinct subsequences of T in S.
# A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters 
# without disturbing the relative positions of the remaining characters. (ie, "ACE" is a subsequence of "ABCDE" while "AEC" is not).
# Here is an example:
# S = "rabbbit", T = "rabbit"
# Return 3.
class Solution:
	def numDistinct(self, S, T):
		lenS, lenT = len(S), len(T)
		dp = [[0 for j in xrange(lenT+1)] for i in xrange(lenS+1)]
		for i in xrange(lenS+1): # empty string is a subsequence of any string
			dp[i][0] = 1
		for i in xrange(1, lenS+1):
			for j in xrange(1, min(i+1,lenT+1)):
				dp[i][j] = dp[i-1][j-1]+dp[i-1][j] if S[i-1]==T[j-1] else dp[i-1][j]
		return dp[lenS][lenT]

# 116 Populating Next Right Pointers in Each Node
# For example,
# Given the following perfect binary tree,
#		  1
#		/  \
#	   2	3
#	  / \  / \
#	 4  5  6  7
# After calling your function, the tree should look like:
#		  1 -> NULL
#		/  \
#	   2 -> 3 -> NULL
#	  / \  / \
#	 4->5->6->7 -> NULL
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
		self.next = None
class Solution:
	def connect(self, root):
		if root == None: return
		visitedNum = 0
		queue = collections.deque()
		queue.append(root)
		while queue:
			curr = queue.popleft()
			visitedNum += 1
			if curr.left:
				queue.append(curr.left)
				queue.append(curr.right)
			curr.next = None if visitedNum & (visitedNum + 1) == 0 else queue[0]

# 117 Populating Next Right Pointers in Each Node II
# Follow up for problem "Populating Next Right Pointers in Each Node".
# What if the given tree could be any binary tree? Would your previous solution still work?
# Note: You may only use constant extra space.
# For example,
# Given the following binary tree,
#		  1
#		/  \
#	   2	3
#	  / \	\
#	 4   5	7
# After calling your function, the tree should look like:
#		  1 -> NULL
#		/  \
#	   2 -> 3 -> NULL
#	  / \	\
#	 4-> 5 -> 7 -> NULL
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
		self.next = None
class Solution:
	def connect(self, root):
		curr = root
		while curr:
			firstNodeInNextLevel = None
			prev = None
			while curr:
				if not firstNodeInNextLevel:
					firstNodeInNextLevel = curr.left if curr.left else curr.right
				if curr.left:
					if prev: prev.next = curr.left
					prev = curr.left
				if curr.right:
					if prev: prev.next = curr.right
					prev = curr.right
				curr = curr.next
			curr = firstNodeInNextLevel # turn to next level

# 118 Pascal's Triangle
# Given numRows, generate the first numRows of Pascal's triangle.
# For example, given numRows = 5,
# Return
# [
#	  [1],
#	 [1,1],
#	[1,2,1],
#   [1,3,3,1],
#  [1,4,6,4,1]
# ]
class Solution:
	def generate(self, numRows):
		result = []
		for i in xrange(numRows):
			result.append([])
			for j in xrange(i+1):
				if j in (0, i):
					result[i].append(1)
				else:
					result[i].append(result[i-1][j-1]+result[i-1][j])
		return result

# 119 Pascal's Triangle II
# Given an index k, return the kth row of the Pascal's triangle.
# For example, given k = 3,
# Return [1,3,3,1].
class Solution:
	def getRow(self, rowIndex):
		result = [1]
		for i in range(1, rowIndex+1):
			result = [1]+[result[j-1]+result[j] for j in range(1,i)]+[1]
		return result

# 120 Triangle
# Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.
# For example, given the following triangle
# [
#	  [2],
#	 [3,4],
#	[6,5,7],
#  [4,1,8,3]
# ]
class Solution:
	def minimumTotal(self, triangle):
		length = len(triangle)
		dp = [0 for i in xrange(length)]
		for row in triangle:
			oldDp = dp[:]
			for i in xrange(len(row)):
				if i == 0: 
					dp[i] = oldDp[i]+row[i]
				elif i == len(row)-1:
					dp[i] = oldDp[i-1]+row[i]
				else:
					dp[i] = min(oldDp[i],oldDp[i-1])+row[i]
		return min(dp)

# 121 Best Time to Buy and Sell Stock
# Say you have an array for which the ith element is the price of a given stock on day i.
# If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.
class Solution:
	def maxProfit(self, prices):
		if not prices: return 0 # prices is empty
		maxProfit = 0
		minPrice = prices[0]
		for currPrice in prices:
			minPrice = min(minPrice, currPrice)
			maxProfit = max(maxProfit, currPrice-minPrice)
		return maxProfit

# 122 Best Time to Buy and Sell Stock II
# Say you have an array for which the ith element is the price of a given stock on day i.
# Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
class Solution:
	def maxProfit(self, prices):
		if not prices: return 0 # prices is empty
		maxProfit = 0
		for i in xrange(1,len(prices)):
			if prices[i] > prices[i-1]:
				maxProfit += prices[i]-prices[i-1]
		return maxProfit

# 123 Best Time to Buy and Sell Stock III
# Say you have an array for which the ith element is the price of a given stock on day i.
# Design an algorithm to find the maximum profit. You may complete at most two transactions.
# Note: You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
class Solution:
	def maxProfit(self, prices):
		if not prices: return 0 # prices is empty
		maxProfitForward = []
		minPrice = prices[0]
		maxProfit = -1
		for currPrice in prices:
			minPrice = min(minPrice, currPrice)
			maxProfit = max(maxProfit, currPrice-minPrice)
			maxProfitForward.append(maxProfit)
		maxProfitBackward = []
		maxPrice = prices[-1]
		maxProfit = -1
		for currPrice in reversed(prices):
			maxPrice = max(maxPrice, currPrice)
			maxProfit = max(maxProfit, maxPrice-currPrice)
			maxProfitBackward.insert(0, maxProfit)
		maxProfit = maxProfitForward[-1] # 0 or 1 transaction
		for i in xrange(len(prices)-1): # 2 transactions
			maxProfit = max(maxProfit, maxProfitForward[i]+maxProfitBackward[i+1])
		return maxProfit

# 124 Binary Tree Maximum Path Sum
# Given a binary tree, find the maximum path sum.
# The path may start and end at any node in the tree.
# For example:
# Given the below binary tree,
#		1
#	   / \
#	  2   3
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	deep = 0
	def maxPathSum(self, root):
		if root == None:
			return 0
		if Solution.deep == 0: # Solution.maxSum is set to infinitesimal every time when a new test case starts
			Solution.maxSum = -10 ** 10
		Solution.deep += 1
		vLeft = self.maxPathSum(root.left)
		vRight = self.maxPathSum(root.right)
		Solution.deep -= 1
		Solution.maxSum = max(root.val+vLeft+vRight, Solution.maxSum)
		if Solution.deep == 0:
			return Solution.maxSum
		return max(root.val+vLeft, root.val+vRight, 0)

# 125 Valid Palindrome 
# Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
# For example,
# "A man, a plan, a canal: Panama" is a palindrome.
# "race a car" is not a palindrome.
class Solution:
	def isPalindrome(self, s):
		newS = []
		for i in s:
			if '0' <= i <= '9' or 'a' <= i <= 'z': newS.append(i)
			elif 'A' <= i <= 'Z': newS.append(chr(ord(i)-ord('A')+ord('a')))
		return newS == newS[::-1]

# 126 Word Ladder II
# Given two words (start and end), and a dictionary, find all shortest transformation sequence(s) from start to end, such that:
# Only one letter can be changed at a time
# Each intermediate word must exist in the dictionary
# For example,
# Given:
# start = "hit"
# end = "cog"
# dict = ["hot","dot","dog","lot","log"]
# Return
#   [
#	 ["hit","hot","dot","dog","cog"],
#	 ["hit","hot","lot","log","cog"]
#   ]
# Note:
# All words have the same length.
# All words contain only lowercase alphabetic characters.
class Solution:
	def findLadders(self, start, end, dict):
		dict.update([start,end])
		result, cur, visited, found, trace = [], [start], set([start]), False, {word:[] for word in dict}  
		while cur and not found:
			for word in cur:
				visited.add(word)
			next = set([])
			for word in cur:
				for i in xrange(len(word)):
					for j in 'abcdefghijklmnopqrstuvwxyz':
						candidate = word[:i]+j+word[i+1:]
						if candidate not in visited and candidate in dict:
							if candidate == end:
								found = True
							next.add(candidate)
							trace[candidate].append(word)
			cur = next
		if found:
			self.backtrack(result, trace, [], end)
		return result
	def backtrack(self, result, trace, path, word):
		if not trace[word]:
			result.append([word] + path)
		else:
			for prev in trace[word]:
				self.backtrack(result, trace, [word] + path, prev)

# 127 Word Ladder
# Given two words (beginWord and endWord), and a dictionary, find the length of shortest transformation sequence from beginWord to endWord, such that:
# Only one letter can be changed at a time
# Each intermediate word must exist in the dictionary
# For example,
# Given:
# start = "hit"
# end = "cog"
# dict = ["hot","dot","dog","lot","log"]
# As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
# return its length 5.
# Note:
# Return 0 if there is no such transformation sequence.
# All words have the same length.
# All words contain only lowercase alphabetic characters.
class Solution:
	def ladderLength(self, start, end, dict):
		dict.add(end)
		wordLen = len(start)
		queue = collections.deque([(start, 1)])
		while queue:
			curr = queue.popleft()
			currWord = curr[0]; currLen = curr[1]
			if currWord == end: return currLen
			for i in xrange(wordLen):
				part1 = currWord[:i]; part2 = currWord[i+1:]
				for j in 'abcdefghijklmnopqrstuvwxyz':
					if currWord[i] != j:
						nextWord = part1 + j + part2
						if nextWord in dict:
							queue.append((nextWord, currLen + 1))
							dict.remove(nextWord)
		return 0

# 128 Longest Consecutive Sequence 
# Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
# For example,
# Given [100, 4, 200, 1, 3, 2],
# The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
# Your algorithm should run in O(n) complexity.
class Solution:
	def longestConsecutive(self, num):
		dict = {x:False for x in num} # False means not visited
		maxLen = -1
		for i in dict:
			if dict[i] == False:
				curr = i+1; len1 = 0
				while curr in dict and dict[curr] == False:
					len1 += 1; dict[curr] = True; curr += 1
				curr = i-1; len2 = 0
				while curr in dict and dict[curr] == False:
					len2 += 1; dict[curr] = True; curr -= 1
				maxLen = max(maxLen, 1+len1+len2)
		return maxLen

# 129 Sum Root to Leaf Numbers
# Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
# An example is the root-to-leaf path 1->2->3 which represents the number 123.
# Find the total sum of all root-to-leaf numbers.
# For example,
#	 1
#	/ \
#  2   3
# The root-to-leaf path 1->2 represents the number 12.
# The root-to-leaf path 1->3 represents the number 13.
# Return the sum = 12 + 13 = 25.
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def sumNumbers(self, root):
		if root == None: return 0
		if root.left == None and root.right == None: return root.val
		Solution.res = 0
		if root.left: self.getSum(root.left, root.val)
		if root.right: self.getSum(root.right, root.val)
		return Solution.res
	def getSum(self, root, valFromParent):
		if root == None: return
		if root.left == None and root.right == None:
			Solution.res += 10 * valFromParent + root.val
		self.getSum(root.left, 10 * valFromParent + root.val)
		self.getSum(root.right, 10 * valFromParent + root.val)
		return

# 130 Surrounded Regions
# Given a 2D board containing 'X' and 'O', capture all regions surrounded by 'X'.
# A region is captured by flipping all 'O's into 'X's in that surrounded region.
# For example,
# X X X X
# X O O X
# X X O X
# X O X X
# After running your function, the board should be:
# X X X X
# X X X X
# X X X X
# X O X X
class Solution:
	def solve(self, board):
		if board == []: return []
		lineNum, colNum = len(board), len(board[0])
		queue = collections.deque()
		visited = [[False for j in xrange(colNum)] for i in xrange(lineNum)]
		for i in xrange(colNum): 
			if board[0][i] == 'O': queue.append((0, i))
			if board[lineNum-1][i] == 'O': queue.append((lineNum-1, i))
		for i in xrange(1, lineNum-1):
			if board[i][0] == 'O': queue.append((i, 0))
			if board[i][colNum-1] == 'O': queue.append((i, colNum-1))
		while queue:
			t = queue.popleft()
			if board[t[0]][t[1]] == 'O': board[t[0]][t[1]] = '$'
			visited[t[0]][t[1]] = True
			if t[0]+1 < lineNum and board[t[0]+1][t[1]] == 'O' and visited[t[0]+1][t[1]] == False: 
				queue.append((t[0]+1, t[1]))
			if t[0]-1 >= 0 and board[t[0]-1][t[1]] == 'O' and visited[t[0]-1][t[1]] == False: 
				queue.append((t[0]-1, t[1]))
			if t[1]+1 < colNum and board[t[0]][t[1]+1] == 'O' and visited[t[0]][t[1]+1] == False: 
				queue.append((t[0], t[1]+1))
			if t[1]-1 >= 0 and board[t[0]][t[1]-1] == 'O' and visited[t[0]][t[1]-1] == False:
				queue.append((t[0], t[1]-1))
		for i in xrange(lineNum):
			for j in xrange(colNum):
				if board[i][j] == 'O': board[i][j] = 'X'
				if board[i][j] == '$': board[i][j] = 'O'

# 131 Palindrome Partitioning
# Given a string s, partition s such that every substring of the partition is a palindrome.
# Return all possible palindrome partitioning of s.
# For example, given s = "aab",
# Return
#   [
#	 ["aa","b"],
#	 ["a","a","b"]
#   ]
class Solution:
	def partition(self, s):
		self.s, self.res, self.lenS = s, [], len(s)
		self.isPal = [[False for j in xrange(self.lenS)] for i in xrange(self.lenS)]
		for j in xrange(self.lenS):
			for i in reversed(xrange(j+1)):
				if s[i] == s[j] and (j-i <= 1 or self.isPal[i+1][j-1] == True):
					self.isPal[i][j] = True # True means substring from s[i](included) to s[j](included) is palindrome
		self.dfs(0, [])
		return self.res
	def dfs(self, start, L):
		if start == self.lenS: 
			self.res.append(L)
			return
		for end in xrange(start, self.lenS):
			if self.isPal[start][end] == True:
				self.dfs(end+1, L[:]+[self.s[start:end+1]]) # make a copy of list L and add a substring

# 132 Palindrome Partitioning II
# Given a string s, partition s such that every substring of the partition is a palindrome.
# Return the minimum cuts needed for a palindrome partitioning of s.
# For example, given s = "aab",
# Return 1 since the palindrome partitioning ["aa","b"] could be produced using 1 cut.
class Solution:
	def minCut(self, s):
		lenS = len(s)
		isPal = [[False for j in xrange(lenS)] for i in xrange(lenS)]
		minPalNum = [i+1 for i in xrange(lenS)]
		for j in xrange(lenS):
			for i in reversed(xrange(j+1)):
				if s[i] == s[j] and (j-i <= 1 or isPal[i+1][j-1] == True):
					isPal[i][j] = True
					minPalNum[j] = min(minPalNum[j], minPalNum[i-1]+1) if i > 0 else min(minPalNum[j], 1) # i == 0
		return minPalNum[lenS - 1] - 1

# 133 Clone Graph
# Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.
# OJ's undirected graph serialization: Nodes are labeled uniquely.
# We use # as a separator for each node, and, as a separator for node label and each neighbor of the node.
# As an example, consider the serialized graph {0,1,2#1,2#2,2}.
# The graph has a total of three nodes, and therefore contains three parts as separated by #.
# First node is labeled as 0. Connect node 0 to both nodes 1 and 2.
# Second node is labeled as 1. Connect node 1 to node 2.
# Third node is labeled as 2. Connect node 2 to node 2 (itself), thus forming a self-cycle.
# Visually, the graph looks like the following:
#		1
#	   / \
#	  /   \
#	 0 --- 2
#		  / \
#		  \_/
class UndirectedGraphNode:
	def __init__(self, x):
		self.label = x
		self.neighbors = []
class Solution:
	def cloneGraph(self, node):
		if node == None: return None
		newNodes = {node.label: UndirectedGraphNode(node.label)}
		q = collections.deque(); q.append(node)
		newQ = collections.deque(); newQ.append(newNodes[node.label])
		visited = set([node]) # A node will be visited as long as it's enqueued
		while q:
			currNode = q.popleft(); newCurrNode = newQ.popleft()
			for n in currNode.neighbors:
				if n.label not in newNodes:
					newNodes[n.label] = UndirectedGraphNode(n.label)
				newCurrNode.neighbors.append(newNodes[n.label])
				if n not in visited: 
					q.append(n); newQ.append(newNodes[n.label])
					visited.add(n) # A node will be visited as long as it's enqueued
		return newNodes[node.label]

# 134 Gas Station
# There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
# You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). 
# You begin the journey with an empty tank at one of the gas stations.
# Return the starting gas station's index if you can travel around the circuit once, otherwise return -1.
class Solution:
	def canCompleteCircuit(self, gas, cost):
		num = len(gas)
		Sum = total = 0
		k = -1
		for i in xrange(num):
			Sum += gas[i]-cost[i]
			total += gas[i]-cost[i]
			if Sum < 0:
				k = i
				Sum = 0
		return k+1 if total >= 0 else -1

# 135 Candy
# There are N children standing in a line. Each child is assigned a rating value.
# You are giving candies to these children subjected to the following requirements:
# Each child must have at least one candy.
# Children with a higher rating get more candies than their neighbors.
# What is the minimum candies you must give?
class Solution:
	def candy(self, ratings):
		length = len(ratings)
		candy = [1 for i in xrange(length)]
		for i in xrange(length - 1):
			if ratings[i+1] > ratings[i] and candy[i+1] <= candy[i]:
				candy[i+1] = candy[i] + 1
		for i in reversed(xrange(1, length)):
			if ratings[i-1] > ratings[i] and candy[i-1] <= candy[i]:
				candy[i-1] = candy[i] + 1
		return sum(candy)

# 136 Single Number
# Given an array of integers, every element appears twice except for one. Find that single one.
# Note: Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
class Solution:
	def singleNumber(self, A):
		res = 0
		for i in A: res ^= i
		return res

# 137 Single Number II
# Given an array of integers, every element appears three times except for one. Find that single one.
# Note: Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
class Solution:
	def singleNumber(self, A):
		bit = [0 for i in xrange(32)]
		for number in A:
			for i in xrange(32):
				if (1<<i)&number == 1<<i: bit[i] += 1
		res = 0
		if bit[31] % 3 == 0: # target number is positive
			for i in xrange(31):
				if bit[i] % 3 == 1: res += 1 << i
		else: # target number is negative
			for i in xrange(31):
				if bit[i] % 3 == 0: res += 1 << i
			res = -(res+1) # now res = -(11..11 - y + 1) = x
		return res

# 138 Copy List with Random Pointer
# A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.
# Return a deep copy of the list.
class RandomListNode:
	def __init__(self, x):
		self.label = x
		self.next = None
		self.random = None
class Solution:
	def copyRandomList(self, head):
		if head == None: return None
		# change N1->N2->N3... to N1->newN1->N2->newN2->N3->newN3...
		p, num = head, 0
		while p:
			t = RandomListNode(p.label)
			t.next, p.next, t.random = p.next, t, p.random
			p, num = t.next, num+1
		# let newNode.random point to correct node
		p = head.next
		for i in xrange(num):
			if p.random: p.random = p.random.next
			if p and p.next: p = p.next.next
		# restore original list and get new list
		p1, p2, newHead = head, head.next, head.next
		for i in xrange(num-1):
			p1.next, p2.next = p1.next.next, p2.next.next
			p1, p2 = p1.next, p2.next
		p1.next, p2.next = None, None
		return newHead

# 139 Word Break
# Given a string s and a dictionary of words dict, determine if s can be segmented into a space-separated sequence of one or more dictionary words.
# For example, given
# s = "leetcode",
# dict = ["leet", "code"].
# Return true because "leetcode" can be segmented as "leet code".
class Solution:
	def wordBreak(self, s, wordDict):
		d, dp = {w:True for w in wordDict}, [True]
		for i in range(len(s)):
			dp.append(any(dp[j] and s[j:i+1] in d for j in range(i+1)))
		return dp[-1]

# 140 Word Break II
# Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where each word is a valid dictionary word.
# Return all such possible sentences.
# For example, given
# s = "catsanddog",
# dict = ["cat", "cats", "and", "sand", "dog"].
# A solution is ["cats and dog", "cat sand dog"].
class Solution:
	def check(self, s, dict):
		d, dp = {w:True for w in dict}, [True]
		for i in range(len(s)):
			dp.append(any(dp[j] and s[j:i+1] in d for j in range(i+1)))
		return dp[-1]
	def dfs(self, s, dict, stringlist):
		if self.check(s, dict):
			if len(s) == 0: Solution.res.append(stringlist[1:])
			for i in range(1, len(s)+1):
				if s[:i] in dict:
					self.dfs(s[i:], dict, stringlist+' '+s[:i])
	def wordBreak(self, s, dict):
		Solution.res = []
		self.dfs(s, dict, '')
		return Solution.res

# 141 Linked List Cycle 
# Given a linked list, determine if it has a cycle in it.
# Follow up: Can you solve it without using extra space?
class ListNode:
	def __init__(self, x):
		self.val = x
		self.next = None
class Solution:
	def hasCycle(self, head):
		fastP = slowP = head
		while fastP != None and fastP.next != None:
			fastP = fastP.next.next
			slowP = slowP.next
			if fastP == slowP: return True
		return False

# 142 Linked List Cycle II
# Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
# Follow up: Can you solve it without using extra space?
class ListNode:
	def __init__(self, x):
		self.val = x
		self.next = None
class Solution:
	def detectCycle(self, head):
		if head == None: return None
		fast = slow = head
		hasCycle = False
		while fast != None and fast.next != None:
			fast = fast.next.next
			slow = slow.next
			if fast == slow:
				hasCycle = True
				break
		if not hasCycle: return None
		fast = head
		while fast != slow:
			fast = fast.next
			slow = slow.next
		return fast

# 143 Reorder List
# Given a singly linked list L: L0→L1→…→Ln-1→Ln,
# reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…
# You must do this in-place without altering the nodes' values.
# For example,
# Given {1,2,3,4}, reorder it to {1,4,2,3}.
class ListNode:
	def __init__(self, x):
		self.val = x
		self.next = None
class Solution:
	def reorderList(self, head):
		if head == None or head.next == None: return
		# find middle node
		fast = slow = head
		while True:
			if fast.next: fast = fast.next
			else: break
			if fast.next: fast, slow = fast.next, slow.next
			else: break
		# reverse the second half list
		prev, curr, head2, slow.next = slow.next, slow.next.next, slow.next, None
		while curr:
			prev.next, curr.next, head2 = curr.next, head2, curr 
			curr = prev.next
		# join the first half and the second half
		dummy2 = ListNode(0)
		dummy2.next, head2 = head2, dummy2 # add a dummy node to second half list
		prev1, curr1, prev2, curr2 = head, head.next, head2, head2.next
		while curr2:
			# insert curr2 node to first half list
			prev2.next, prev1.next, curr2.next = curr2.next, curr2, curr1 
			curr2 = prev2.next
			prev1 = curr1
			if prev1 == None: break
			curr1 = prev1.next

# 144 Binary Tree Preorder Traversal
# Given a binary tree, return the preorder traversal of its nodes' values.
# For example:
# Given binary tree {1,#,2,3},
#	1
#	 \
#	  2
#	 /
#	3
# return [1,2,3].
# Note: Recursive solution is trivial, could you do it iteratively?
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def preorderTraversal(self, root):
		res = []
		if root == None:
			return res
		stack = [root]
		while stack:
			root = stack.pop()
			res.append(root.val)
			if root.right:
				stack.append(root.right)
			if root.left:
				stack.append(root.left)
		return res

# 145 Binary Tree Postorder Traversal
# Given a binary tree, return the postorder traversal of its nodes' values.
# For example:
# Given binary tree {1,#,2,3},
#	1
#	 \
#	  2
#	 /
#	3
# return [3,2,1].
# Note: Recursive solution is trivial, could you do it iteratively?
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def postorderTraversal(self, root):
		stack = []; res = []
		pre = None
		while root or stack:
			if root:
				stack.append(root)
				root = root.left
			elif pre == stack[-1].right:
				pre = stack.pop()
				res.append(pre.val)
			else:
				root = stack[-1].right
				pre = None
		return res

# 146 LRU Cache
# Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and set.
# get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
# set(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, 
# it should invalidate the least recently used item before inserting a new item.
class LRUCache:
	def __init__(self, capacity):
		LRUCache.Dict = collections.OrderedDict()
		LRUCache.capacity = capacity
		LRUCache.numItems = 0
	def get(self, key):
		try:
			value = LRUCache.Dict[key]
			del LRUCache.Dict[key]
			LRUCache.Dict[key] = value
			return value
		except:
			return -1
	def set(self, key, value):
		try:
			del LRUCache.Dict[key]
			LRUCache.Dict[key] = value
			return
		except:
			if LRUCache.numItems == LRUCache.capacity:
				LRUCache.Dict.popitem(last=False)
				LRUCache.numItems -= 1
			LRUCache.Dict[key] = value
			LRUCache.numItems += 1
		return
class LRUCache:
	class Node:
		def __init__(self, key, value):
			self.key = key
			self.value = value
			self.prev = None
			self.next = None
	def __init__(self, capacity):
		self.capacity, self.dict = capacity, {}
		self.head, self.tail = self.Node('head', 'head'), self.Node('tail', 'tail')
		self.head.next = self.tail
		self.tail.prev = self.head
	def get(self, key):
		if key not in self.dict:
			return -1
		else:
			self.insertNodeAtFirst(self.unlinkNode(self.dict[key]))
			return self.dict[key].value
	def set(self, key, value):
		if key in self.dict:
			self.insertNodeAtFirst(self.unlinkNode(self.dict[key]))
			self.dict[key].value = value
		else:
			if len(self.dict) >= self.capacity:
				del self.dict[self.unlinkNode(self.tail.prev).key]
			self.dict[key] = self.Node(key, value)
			self.insertNodeAtFirst(self.dict[key])
	def unlinkNode(self, node):
		node.prev.next = node.next
		node.next.prev = node.prev
		node.prev = None
		node.next = None
		return node
	def insertNodeAtFirst(self, node):
		node.prev = self.head
		node.next = self.head.next
		self.head.next.prev = node
		self.head.next = node

# 147 Insertion Sort List
# Sort a linked list using insertion sort.
class ListNode:
	def __init__(self, x):
		self.val = x
		self.next = None
class Solution:
	def insertionSortList(self, head):
		if head is None or self.isSorted(head):
			return head
		dummy = ListNode(float('-inf'))
		dummy.next = head
		sorted_tail, cur = head, head.next
		while cur:
			prev = dummy
			while prev.next.val < cur.val:
				prev = prev.next
			if prev == sorted_tail:
				cur, sorted_tail = cur.next, cur
			else:
				prev.next, cur.next, sorted_tail.next = cur, prev.next, cur.next
				cur = sorted_tail.next
		return dummy.next
	def isSorted(self, head):
		while head and head.next:
			if head.val > head.next.val:
				return False
			head = head.next
		return True

# 148 Sort List
# Sort a linked list in O(n log n) time using constant space complexity.
class Solution:
	def sortList(self, head):
		length, curr = 0, head
		while curr != None:
			length += 1
			curr = curr.next
		return self.mergeSort(head, length)
	def mergeSort(self, head, length):
		# base case
		if length == 1 or length == 0: return head
		# sort the two halves of the list
		prev = curr = head
		for i in xrange(length/2): curr = curr.next
		for i in xrange(length/2-1): prev = prev.next
		prev.next = None
		head1 = self.mergeSort(head, length/2)
		head2 = self.mergeSort(curr, length-length/2)
		# merge sorted halves into one
		if head1.val <= head2.val: 
			newHead = curr = ListNode(head1.val)
			head1 = head1.next
		else: 
			newHead = curr = ListNode(head2.val)
			head2 = head2.next
		while head1 and head2:
			if head1.val <= head2.val:
				curr.next = ListNode(head1.val)
				head1 = head1.next
			else:
				curr.next = ListNode(head2.val)
				head2 = head2.next
			curr = curr.next
		while head1:
			curr.next = ListNode(head1.val)
			head1 = head1.next
			curr = curr.next
		while head2:
			curr.next = ListNode(head2.val)
			head2 = head2.next
			curr = curr.next
		return newHead

# 149 Max Points on a Line
# Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.
class Point:
	def __init__(self, a=0, b=0):
		self.x = a
		self.y = b
class Solution:
	def maxPoints(self, points):
		length = len(points)
		if length < 3: return length
		res = -1
		for i in xrange(length):
			slope = {'inf': 0}
			samePointNum = 1
			for j in xrange(length):
				if i == j: continue
				if points[i].x != points[j].x:
					k = 1.0*(points[i].y-points[j].y)/(points[i].x-points[j].x)
					slope[k] = 1 if k not in slope else slope[k]+1
				elif points[i].y != points[j].y:
					slope['inf'] += 1
				else:
					samePointNum += 1
			res = max(res, max(slope.values())+samePointNum)
		return res

# 150 Evaluate Reverse Polish Notation
# Evaluate the value of an arithmetic expression in Reverse Polish Notation.
# Valid operators are +, -, *, /. Each operand may be an integer or another expression.
# Some examples:
#   ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
#   ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6
class Solution:
	def evalRPN(self, tokens):
		stack = []
		for i in tokens:
			if i not in ('+', '-', '*', '/'):
				stack.append(int(i))
			else:
				op2 = stack.pop()
				op1 = stack.pop()
				if i == '+': stack.append(op1+op2)
				elif i == '-': stack.append(op1-op2)
				elif i == '*': stack.append(op1*op2)
				else: stack.append(int(1.0*op1/op2))
		return stack[0]

# 151 Reverse Words in a String
# For example,
# Given s = "the sky is blue",
# return "blue is sky the".
class Solution:
	def reverseWords(self, s):
		return ' '.join(s.split()[::-1])

# 152 Maximum Product Subarray
# Find the contiguous subarray within an array (containing at least one number) which has the largest product.
# For example, given the array [2,3,-2,4],
# the contiguous subarray [2,3] has the largest product = 6.
class Solution:
	def maxProduct(self, A):
		global_max, local_max, local_min = float("-inf"), 1, 1
		for x in A:
			local_max, local_min = max(x, local_max*x, local_min*x), min(x, local_max*x, local_min*x)
			global_max = max(global_max, local_max)
		return global_max

# 153 Find Minimum in Rotated Sorted Array
# Suppose a sorted array is rotated at some pivot unknown to you beforehand.
# (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
# Find the minimum element.
# You may assume no duplicate exists in the array.
class Solution:
	def findMin(self, num):
		low, high = 0, len(num)-1
		while low < high and num[low] >= num[high]:
			mid = low+(high-low)/2
			if num[mid] >= num[low]:
				low = mid+1
			else:
				high = mid
		return num[low]

# 154 Find Minimum in Rotated Sorted Array II
# Follow up for "Find Minimum in Rotated Sorted Array":
# What if duplicates are allowed?
# Would this affect the run-time complexity? How and why?
# Suppose a sorted array is rotated at some pivot unknown to you beforehand.
# (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
# Find the minimum element.
# The array may contain duplicates.
class Solution:
	def findMin(self, num):
		low, high = 0, len(num) - 1
		while low < high and num[low] >= num[high]:
			mid = low+(high-low)/2
			if num[mid] > num[low]:
				low = mid+1
			elif num[mid] < num[low]:
				high = mid
			else:
				low += 1
		return num[low]

# 155 Min Stack
# Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
# push(x) -- Push element x onto stack.
# pop() -- Removes the element on top of the stack.
# top() -- Get the top element.
# getMin() -- Retrieve the minimum element in the stack.
class MinStack:
	def __init__(self):
		self.stack1 = []
		self.stack2 = []
	def push(self, x):
		self.stack1.append(x)
		if len(self.stack2) == 0 or x <= self.stack2[-1]:
			self.stack2.append(x)
	def pop(self):
		top = self.stack1.pop()
		if top == self.stack2[-1]:
			self.stack2.pop()
	def top(self):
		return self.stack1[-1]
	def getMin(self):
		return self.stack2[-1]

# 156 Binary Tree Upside Down
# Given a binary tree where all the right nodes are either leaf nodes with a sibling (a left node that shares the same parent node) or empty, 
# flip it upside down and turn it into a tree where the original right nodes turned into left leaf nodes. Return the new root.
# For example:
# Given a binary tree {1,2,3,4,5},
#	  1
#	 / \
#   2   3
#  / \
# 4   5
# return the root of the binary tree [4,5,2,#,#,3,1].
#	 4
#   / \
#  5   2
#	  / \
#	 3   1  
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def UpsideDownBinaryTree(root):
		p, parent, parentRight = root, None, None
		while p != None:
			left = p.left
			p.left = parentRight
			parentRight = p.right
			p.right = parent
			parent = p
			p = left
		return parent

# 157 Read N Characters Given Read4
# The API: int read4(char *buf) reads 4 characters at a time from a file.
# The return value is the actual number of characters read. For example, it returns 3 if there is only 3 characters left in the file.
# By using the read4 API, implement the function int read(char *buf, int n) that reads n characters from the file.
# Note: The read function will only be called once for each test case.
class Solution:
	def read(self, buf, n):
		idx = 0
		while True:
			buf4 = [""]*4
			curr = min(read4(buf4), n-idx)  # curr is the number of chars that reads
			for i in xrange(curr):
				buf[idx] = buf4[i]
				idx += 1
			if curr != 4 or idx == n:  # return if it reaches the end of file or reaches n
				return idx

# 158 Read N Characters Given Read4 II – Call multiple times
# Note: The read function may be called multiple times.
def read4(buf):
	global file_content
	i = 0
	while i < len(file_content) and i < 4:
		buf[i] = file_content[i]
		i += 1
	if len(file_content) > 4:
		file_content = file_content[4:]
	else:
		file_content = ""
	return i
class Solution:
	def __init__(self):
		self.buffer_size, self.offset = 0, 0
		self.buffer = [None for _ in xrange(4)]
	def read(self, buf, n):
		read_bytes, eof = 0, False
		while not eof and read_bytes < n:
			if self.buffer_size == 0:
				size = read4(self.buffer)
			else:
				size = self.buffer_size
			if self.buffer_size == 0 and size < 4:
				eof = True
			bytes = min(n-read_bytes, size)
			for i in xrange(bytes):
				buf[read_bytes+i] = self.buffer[self.offset+i]
			self.offset = (self.offset+bytes)%4
			self.buffer_size = size-bytes
			read_bytes += bytes
		return read_bytes

# 159 Longest Substring with At Most Two Distinct Characters
# Given a string S, find the length of the longest substring T that contains at most two distinct characters.
# For example,
# Given S = “eceba”,
# T is “ece” which its length is 3.
class Solution:
	def lengthOfLongestSubstringTwoDistinct(s):
		tail, seen, lenmax = 0, {s[0]:1}, 0
		for head in xrange(1,len(s)):
			if s[head] in seen or len(seen) < 2:
				seen[s[head]] = seen.get(s[head],0)+1
				lenmax = max(lenmax,head-tail+1)
			else:
				seen[s[tail]] -= 1
				if seen[s[tail]] == 0:
					del seen[s[tail]]
				tail += 1
		print lenmax

# 160 Intersection of Two Linked Lists 
# Write a program to find the node at which the intersection of two singly linked lists begins.
# For example, the following two linked lists:
# A:		  a1 → a2
#					↘
#					  c1 → c2 → c3
#					↗			
# B:	 b1 → b2 → b3
# begin to intersect at node c1.
class ListNode:
	def __init__(self, x):
		self.val = x
		self.next = None
class Solution:
	def getIntersectionNode(self, headA, headB):
		curA, curB = headA, headB
		tailA, tailB = None, None
		while curA and curB:
			if curA == curB:
				return curA
			if curA.next:
				curA = curA.next
			elif tailA is None:
				tailA = curA
				curA = headB
			else:
				break
			if curB.next:
				curB = curB.next
			elif tailB is None:
				tailB = curB
				curB = headA
			else:
				break
		return None

# 161 One Edit Distance
# Given two strings S and T, determine if they are both one edit distance apart.
# Hint:
# 1. If | n – m | is greater than 1, we know immediately both are not one-edit distance apart.
# 2. It might help if you consider these cases separately, m == n and m ≠ n.
# 3. Assume that m is always ≤ n, which greatly simplifies the conditional statements. If m > n, we could just simply swap S and T.
# 4. If m == n, it becomes finding if there is exactly one modified operation. If m ≠ n, you do not have to consider the delete operation. Just consider the insert operation in T.
class Solution:
	def isOneEditDistance(s, t):
		m, n = len(s), len(t)
		if m > n:
			return isOneEditDistance(t, s)
		if n - m > 1:
			return False
		i, shift = 0, n-m
		while i < m and s[i] == t[i]:
			i += 1
		if i == m:
			return True if shift == 1 else False
		if shift == 0:
			i += 1
		while i < m and s[i] == t[i+shift]:
			i += 1
		return True if i == m else False

# 162 Find Peak Element
# A peak element is an element that is greater than its neighbors.
# Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.
# The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.
# You may imagine that num[-1] = num[n] = -∞.
# For example, in array [1, 2, 3, 1], 3 is a peak element and your function should return the index number 2.
class Solution:
	def findPeakElement(self, num):
		low, high = 0, len(num)-1
		while low < high:
			mid = (low+high)/2
			if num[mid] < num[mid+1]:
				low = mid+1
			else:
				high = mid
		return low

# 163 Missing Ranges
# Given a sorted integer array where the range of elements are [0, 99] inclusive, return its missing ranges.
# For example, given [0, 1, 3, 50, 75], return [“2”, “4->49”, “51->74”, “76->99”]
class Solution:
	def findMissingRanges(vals):
		vals.insert(0,-1)
		vals.insert(len(vals),100)
		r = []
		for i in range(1,len(vals)):
			d = vals[i]-vals[i-1]
			if d == 1:
				continue
			elif d == 2:
				r.append(str(vals[i-1]+1))
			else:
				r.append(str(vals[i-1]+1)+"->"+str(vals[i]-1))
		return r

# 164 Maximum Gap
# Given an unsorted array, find the maximum difference between the successive elements in its sorted form.
# Try to solve it in linear time/space.
# Return 0 if the array contains less than 2 elements.
# You may assume all elements in the array are non-negative integers and fit in the 32-bit signed integer range.
class Solution:
	def maximumGap(self, nums):
		if len(nums) >= 2:
			maxV, minV = max(nums), min(nums)
			gap, bs = (maxV-minV)/(len(nums)-1), [[] for _ in nums]
			for n in nums:
				bs[(n-minV)/(gap+1)].append(n)
			bs = filter(None, bs)
		return max([min(a)-max(b) for a,b in zip(bs[1:],bs[:-1])]+[gap]) if len(nums)>=2 else 0

# 165 Compare Version Numbers
# Compare two version numbers version1 and version2.
# If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.
# You may assume that the version strings are non-empty and contain only digits and the . character.
# The . character does not represent a decimal point and is used to separate number sequences.
# For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level revision.
# Here is an example of version numbers ordering:
# 0.1 < 1.1 < 1.2 < 13.37
class Solution:
	def compareVersion(self, version1, version2):
		v1, v2 = version1.split("."), version2.split(".")
		if len(v1) > len(v2):
			v2 += ['0' for _ in xrange(len(v1) - len(v2))]
		elif len(v1) < len(v2):
			v1 += ['0' for _ in xrange(len(v2) - len(v1))]
		i = 0
		while i < len(v1):
			if int(v1[i]) > int(v2[i]):
				return 1
			elif int(v1[i]) < int(v2[i]):
				return -1
			else:
				i += 1
		return 0

# 166 Fraction to Recurring Decimal
# Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
# If the fractional part is repeating, enclose the repeating part in parentheses.
# For example,
# Given numerator = 1, denominator = 2, return "0.5".
# Given numerator = 2, denominator = 1, return "2".
# Given numerator = 2, denominator = 3, return "0.(6)".
class Solution:
	def fractionToDecimal(self, numerator, denominator):
		res = ""
		if numerator/denominator<0:
			res += "-"
		elif numerator%denominator==0:
			return str(numerator/denominator)
		numerator, denominator = abs(numerator), abs(denominator)
		res += str(numerator/denominator)+"."
		numerator %= denominator
		i, table = len(res), {}
		while numerator!=0:
			if numerator not in table.keys():
				table[numerator]=i
			else:
				i = table[numerator]
				res = res[:i]+"("+res[i:]+")"
				return res
			numerator = numerator*10
			res += str(numerator/denominator)
			numerator %= denominator
			i += 1
		return res

# 167 Two Sum II – Input array is sorted
# Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
# The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. 
# Please note that your returned answers (both index1 and index2) are not zero-based.
# You may assume that each input would have exactly one solution.
# Input: numbers={2, 7, 11, 15}, target=9
# Output: index1=1, index2=2
class Solution:
	def twoSum(numbers, target):
		h, t = len(numbers)-1, 0
		while h >= t:
			if numbers[t]+numbers[h] == target:
				return [t+1,h+1]
			else:
				if numbers[t]+numbers[h] > target:
					h -= 1
				else:
					t += 1
		return [-1,-1]

# 168 Excel Sheet Column Title
# Given a positive integer, return its corresponding column title as appear in an Excel sheet.
# For example:
#	 1 -> A
#	 2 -> B
#	 3 -> C
#	 ...
#	 26 -> Z
#	 27 -> AA
#	 28 -> AB 
class Solution:
	def convertToTitle(self, num):
		result, dvd = "", num
		while dvd:
			result += chr((dvd-1)%26+ord('A'))
			dvd = (dvd-1)/26
		return result[::-1]

# 169 Majority Element
# Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
# You may assume that the array is non-empty and the majority element always exist in the array.
class Solution:
	def majorityElement(self, num):
		idx, cnt = 0, 1
		for i in xrange(1, len(num)):
			if num[idx] == num[i]:
				cnt += 1
			else:
				cnt -= 1
				if cnt == 0:
					idx = i
					cnt = 1
		return num[idx]

# 170 Two Sum III - Data structure design
# Design and implement a TwoSum class. It should support the following operations: add and find.
# add - Add the number to an internal data structure.
# find - Find if there exists any pair of numbers which sum is equal to the value.
# For example,
# add(1); add(3); add(5);
# find(4) -> true
# find(7) -> false
class TwoSum:
	def __init__(self):
		self.lookup = {}
	def add(self, number):
		if number in self.lookup:
			self.lookup[number] += 1
		else:
			self.lookup[number] = 1
	def find(self, value):
		for key in self.lookup:
			num = value - key
			if num in self.lookup and (num != key or self.lookup[key]>1):
				return True
		return False

# 171 Excel Sheet Column Number
# Related to question Excel Sheet Column Title
# Given a column title as appear in an Excel sheet, return its corresponding column number.
# For example:
#	 A -> 1
#	 B -> 2
#	 C -> 3
#	 ...
#	 Z -> 26
#	 AA -> 27
#	 AB -> 28 
class Solution:
	def titleToNumber(self, s):
		result = 0
		for i in xrange(len(s)):
			result *= 26
			result += ord(s[i])-ord('A')+1
		return result

# 172 Factorial Trailing Zeroes
# Given an integer n, return the number of trailing zeroes in n!.
# Note: Your solution should be in logarithmic time complexity.
class Solution:
	def trailingZeroes(self, n):
		result = 0
		while n > 0:
			result += n/5
			n /= 5
		return result

# 173 Binary Search Tree Iterator
# Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.
# Calling next() will return the next smallest number in the BST.
# Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class BSTIterator:
	# @param root, a binary search tree's root node
	def __init__(self, root):
		self.stack = []
		self.cur = root
	# @return a boolean, whether we have a next smallest number
	def hasNext(self):
		return self.stack or self.cur
	# @return an integer, the next smallest number
	def next(self):
		while self.cur:
			self.stack.append(self.cur)
			self.cur = self.cur.left
		self.cur = self.stack.pop()
		node = self.cur
		self.cur = self.cur.right
		return node.val

# 174 Dungeon Game
# The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms laid out in a 2D grid. 
# Our valiant knight (K) was initially positioned in the top-left room and must fight his way through the dungeon to rescue the princess.The knight has an initial health point represented by a positive integer. 
# If at any point his health point drops to 0 or below, he dies immediately. Some of the rooms are guarded by demons, so the knight loses health (negative integers) upon entering these rooms; 
# other rooms are either empty (0's) or contain magic orbs that increase the knight's health (positive integers).
# In order to reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.
class Solution:
	def calculateMinimumHP(self, dungeon):
		DP = [float("inf") for _ in dungeon[0]]
		DP[-1] = 1
		for i in reversed(xrange(len(dungeon))):
			DP[-1] = max(DP[-1]-dungeon[i][-1], 1)
			for j in reversed(xrange(len(dungeon[i])-1)):
				min_HP_on_exit = min(DP[j], DP[j+1])
				DP[j] = max(min_HP_on_exit-dungeon[i][j], 1)
		return DP[0]

# 179 Largest Number
# Given a list of non negative integers, arrange them such that they form the largest number.
# For example, given [3, 30, 34, 5, 9], the largest formed number is 9534330.
# Note: The result may be very large, so you need to return a string instead of an integer.
class Solution:
	def largestNumber(self, num):
		num = [str(x) for x in num]
		num.sort(cmp=lambda x,y:cmp(y+x,x+y))
		largest = ''.join(num)
		return largest.lstrip('0') or '0'

# 186 Reverse Words in a String II
# Given an input string, reverse the string word by word. A word is defined as a sequence of non-space characters.
# The input string does not contain leading or trailing spaces and the words are always separated by a single space.
# For example,
# Given s = "the sky is blue",
# return "blue is sky the".
# Could you do it in-place without allocating extra space?
class Solution:
	def reverseWords(self, s):
		self.reverse(s, 0, len(s))
		i = 0
		for j in xrange(len(s)+1):
			if j == len(s) or s[j] == ' ':
				self.reverse(s, i, j)
				i = j + 1
	def reverse(self, s, begin, end):
		for i in xrange((end-begin)/2):
			s[begin+i], s[end-1-i] = s[end-1-i], s[begin+i]

# 187 Repeated DNA Sequences
# All DNA is composed of a series of nucleotides abbreviated as A, C, G, and T, for example: "ACGAATTCCG". 
# When studying DNA, it is sometimes useful to identify repeated sequences within the DNA.
# Write a function to find all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule.
# For example,
# Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",
# Return:
# ["AAAAACCCCC", "CCCCCAAAAA"].
class Solution:
	def findRepeatedDnaSequences(self, s):
		res, dict, rolling_hash = [], {}, 0
		for i in xrange(len(s)):
			rolling_hash = rolling_hash<<3&0x3fffffff|ord(s[i])&7
			if dict.get(rolling_hash) is None:
				dict[rolling_hash] = True
			else:
				if dict[rolling_hash]:
					res.append(s[i-9:i+1])
					dict[rolling_hash] = False
		return res

# 188 Best Time to Buy and Sell Stock IV
# Say you have an array for which the ith element is the price of a given stock on day i.
# Design an algorithm to find the maximum profit. You may complete at most k transactions.
# You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
class Solution:
	def maxProfit(self, k, prices):
		length = len(prices)
		if length < 2: return 0
		max_profit = 0
		if k >= length/2:
			for i in xrange(1,length):
				max_profit += max(prices[i]-prices[i-1],0)
			return max_profit
		max_global = [[0]*length for _ in xrange(k+1)]
		max_local = [[0]*length for _ in xrange(k+1)]
		for j in xrange(1,length):
			cur_profit = prices[j]-prices[j-1] # variable introduced by the current day transaction
			for i in xrange(1,k+1):
				max_local[i][j] = max(max_global[i-1][j-1]+max(cur_profit,0), max_local[i][j-1]+cur_profit)
				max_global[i][j] = max(max_global[i][j-1], max_local[i][j])
		return max_global[k][-1] # the last day, the last transaction

# 189 Rotate Array
# Rotate an array of n elements to the right by k steps.
# For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].
class Solution:
	def rotate(self, nums, k):
		k %= len(nums)
		self.reverse(nums, 0, len(nums))
		self.reverse(nums, 0, k)
		self.reverse(nums, k, len(nums))
	def reverse(self, nums, start, end):
		while start < end:
			nums[start], nums[end-1] = nums[end-1], nums[start]
			start += 1
			end -= 1

# 190 Reverse Bits
# Reverse bits of a given 32 bits unsigned integer.
# For example, given input 43261596 (represented in binary as 00000010100101000001111010011100), return 964176192 (represented in binary as 00111001011110000010100101000000).
class Solution:
	def reverseBits(self, n):
		result = 0
		for i in xrange(32):
			result <<= 1
			result |= n & 1
			n >>= 1
		return result

# 191 Number of 1 Bits 
# Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).
# For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.
class Solution:
	def hammingWeight(self, n):
		result = 0
		while n:
			n &= n - 1
			result += 1
		return result

# 198 House Robber
# You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing 
# each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
# Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
# f(0) = nums[0]
# f(1) = max(num[0], num[1])
# f(k) = max(f(k-2)+nums[k], f(k-1))
class Solution:
	def rob(self, nums):
		last, now = 0, 0
		for i in nums: 
			last, now = now, max(last+i,now)
		return now

# 199 Binary Tree Right Side View
# Given a binary tree, imagine yourself standing on the right side of it, 
# return the values of the nodes you can see ordered from top to bottom.
# For example:
# Given the following binary tree,
#	1		<---
#  / \
# 2   3	  <---
#  \   \
#   5   4	<---
# You should return [1, 3, 4].
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None
class Solution:
	def rightSideView(self, root):
		result = []
		self.rightSideViewDFS(root, 1, result)
		return result
	def rightSideViewDFS(self, node, depth, result):
		if not node: return
		if depth > len(result):
			result.append(node.val)
		self.rightSideViewDFS(node.right, depth+1, result)
		self.rightSideViewDFS(node.left, depth+1, result)

# 200 Number of Islands
# Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. 
# An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. 
# Example 1:
# 11110
# 11010
# 11000
# 00000
# Answer: 1
# Example 2:
# 11000
# 11000
# 00100
# 00011
# Answer: 3
class Solution:
	def numIslands(self, grid):
		if grid == []: return 0
		row, col = len(grid), len(grid[0])
		used = [[False for j in xrange(col)] for i in xrange(row)]
		count = 0
		for i in xrange(row):
			for j in xrange(col):
				if grid[i][j] == '1' and not used[i][j]:
					self.dfs(grid, used, row, col, i, j)
					count += 1
		return count
	def dfs(self, grid, used, row, col, x, y):
		if grid[x][y] == '0' or used[x][y]:
			return 0
		used[x][y] = True
		if x != 0:
			self.dfs(grid, used, row, col, x-1, y)
		if x != row-1:
			self.dfs(grid, used, row, col, x+1, y)
		if y != 0:
			self.dfs(grid, used, row, col, x, y-1)
		if y != col-1:
			self.dfs(grid, used, row, col, x, y+1)

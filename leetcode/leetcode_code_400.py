# -*- coding: utf-8 -*- 

# # Remove Invalid Parentheses
# # Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.
# def removeInvalidParentheses(self, s):
#     def isvalid(s):
#         ctr = 0
#         for c in s:
#             if c == '(':
#                 ctr += 1
#             elif c == ')':
#                 ctr -= 1
#                 if ctr < 0:
#                     return False
#         return ctr == 0
#     level = {s}
#     while True:
#         valid = filter(isvalid, level)
#         if valid:
#             return valid
#         level = {s[:i] + s[i+1:] for s in level for i in range(len(s))}

# # Smallest Rectangle Enclosing Black Pixels
# # class Solution(object):
#     def minArea(self, image, x, y): # Binary Search
#         top = self.searchRows(image, 0, x, True)
#         bottom = self.searchRows(image, x + 1, len(image), False)
#         left = self.searchColumns(image, 0, y, top, bottom, True)
#         right = self.searchColumns(image, y + 1, len(image[0]), top, bottom, False)
#         return (right - left) * (bottom - top)
#     def searchRows(self, image, i, j, opt):
#         while i != j:
#             m = (i + j) / 2
#             if any(p == '1' for p in image[m]) == opt:
#                 j = m
#             else:
#                 i = m + 1
#         return i
#     def searchColumns(self, image, i, j, top, bottom, opt):
#         while i != j:
#             m = (i + j) / 2
#             if any(image[k][m] == '1' for k in xrange(top, bottom)) == opt:
#                 j = m
#             else:
#                 i = m + 1
#         return i

# # Range Sum Query - Immutable
# # Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
# class NumArray(object):
#     def __init__(self, nums):
#         size = len(nums)
#         self.sums = [0] * (size + 1)
#         for x in range(size):
#             self.sums[x + 1] += self.sums[x] + nums[x]
#     def sumRange(self, i, j):
#         return self.sums[j + 1] - self.sums[i]

# # Range Sum Query 2D - Immutable
# # Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
# class NumMatrix(object):
#     def __init__(self, matrix):
#         m = len(matrix)
#         n = len(matrix[0]) if m else 0
#         self.sums = [[0] * (n + 1) for x in range(m + 1)]
#         for x in range(1, m + 1):
#             rowSum = 0
#             for y in range(1, n + 1):
#                 self.sums[x][y] += rowSum + matrix[x - 1][y - 1]
#                 if x > 1:
#                     self.sums[x][y] += self.sums[x - 1][y]
#                 rowSum += matrix[x - 1][y - 1]
#     def sumRegion(self, row1, col1, row2, col2):
#         return self.sums[row2 + 1][col2 + 1] + self.sums[row1][col1] \
#                  - self.sums[row1][col2 + 1] - self.sums[row2 + 1][col1]

# # Number of Islands II
# # Given a n,m which means the row and column of the 2D matrix and an array of pair A( size k). Originally, the 2D matrix is all 0 which means there is only sea in the matrix. 
# # The list pair has k operator and each operator has two integer A[i].x, A[i].y means that you can change the grid matrix[A[i].x][A[i].y] from sea to island. 
# # Return how many island are there in the matrix after each operator.
# def numIslands2(m, n, positions):
#     def find(x):
#         if parent[x] != x:
#             parent[x] = find(parent[x])
#         return parent[x]
#     def union(a, b):
#         x, y = find(a), find(b)
#         if x != y:
#             if rank[x] < rank[y]:
#                 parent[x] = y
#             else:
#                 parent[y] = x
#                 rank[x] += 1 if rank[x] == rank[y] else 0
#         return x != y
#     idx = lambda x, y: x * n + y
#     parent = range(idx(m, n))
#     rank = [0] * idx(m, n)
#     matrix = [[0] * n for _ in xrange(m)]
#     ret = []
#     for a, b in positions:
#         matrix[a][b] = 1
#         cnt = ret[-1] + 1 if ret else 1
#         for x, y in (a+1, b), (a-1, b), (a, b+1), (a, b-1):
#             if 0 <= x < m and 0 <= y < n and matrix[x][y] and union(idx(a, b), idx(x, y)):
#                 cnt -= 1
#         ret.append(cnt)
#     return ret

# # Additive Number
# # Additive number is a string whose digits can form additive sequence.
# class Solution(object):
#     def isAdditiveNumber(self, num):
#         length = len(num)
#         for i in range(1, length/2+1):
#             for j in range(1, (length-i)/2 + 1):
#                 first, second, others = num[:i], num[i:i+j], num[i+j:]
#                 if self.isValid(first, second, others):
#                     return True
#         return False
#     def isValid(self, first, second, others):
#         if ((len(first) > 1 and first[0] == "0") or
#                 (len(second) > 1 and second[0] == "0")):
#             return False
#         sum_str = str(int(first) + int(second))
#         if sum_str == others:
#             return True
#         elif others.startswith(sum_str):
#             return self.isValid(second, sum_str, others[len(sum_str):])
#         else:
#             return False

# # Range Sum Query - Mutable
# # Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
# # The update(i, val) function modifies nums by updating the element at index i to val.
# class NumArray(object):
#     def __init__(self, nums):
#         self.n = len(nums)
#         self.a, self.c = nums, [0] * (self.n + 1)
#         for i in range(self.n):
#             k = i + 1
#             while k <= self.n:
#                 self.c[k] += nums[i]
#                 k += (k & -k)
#     def update(self, i, val):
#         diff, self.a[i] = val - self.a[i], val
#         i += 1
#         while i <= self.n:
#             self.c[i] += diff
#             i += (i & -i)
#     def sumRange(self, i, j):
#         res, j = 0, j + 1
#         while j:
#             res += self.c[j]
#             j -= (j & -j)
#         while i:
#             res -= self.c[i]
#             i -= (i & -i)
#         return res

# # Range Sum Query 2D - Mutable
# class NumMatrix(object):
# def __init__(self, matrix):
#     if not matrix:
#         return
#     self.m, self.n = len(matrix), len(matrix[0])
#     self.matrix, self.bit = [[0]*(self.n) for _ in range(self.m)], [[0]*(self.n+1) for _ in range(self.m+1)]
#     for i in range(self.m):
#         for j in range(self.n):
#             self.update(i, j, matrix[i][j])
# def update(self, row, col, val):
#     diff, self.matrix[row][col], i = val-self.matrix[row][col], val, row+1
#     while i <= self.m:
#         j = col+1
#         while j <= self.n:
#             self.bit[i][j] += diff
#             j += (j & -j)
#         i += (i & -i)
# def sumRegion(self, row1, col1, row2, col2):
#     return self.sumCorner(row2, col2) + self.sumCorner(row1-1, col1-1) - self.sumCorner(row1-1, col2) - self.sumCorner(row2, col1-1)
# def sumCorner(self, row, col):
#     res, i = 0, row+1
#     while i:
#         j = col+1
#         while j:
#             res += self.bit[i][j]
#             j -= (j & -j)
#         i -= (i & -i)
#     return res

# # Best Time to Buy and Sell Stock with Cooldown
# # You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:
# # You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
# # After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
# # notHold -- buy --> hold
# # hold -- sell --> cooldown
# # hold -- do nothing --> hold
# # notHold -- do nothing --> notHold
# # cooldown -- do nothing --> notHold
# def maxProfit(self, prices):
#     notHold, notHold_cooldown, hold = 0, float('-inf'), float('-inf')
#     for p in prices:
#         hold, notHold, notHold_cooldown = max(hold, notHold - p), max(notHold, notHold_cooldown), hold + p
#     return max(notHold, notHold_cooldown)

# # Minimum Height Trees
# # For a undirected graph with tree characteristics, we can choose any node as the root. 
# # The result graph is then a rooted tree. Among all possible rooted trees, those with minimum height are called minimum height trees (MHTs). 
# class Solution(object):
#     def findMinHeightTrees(self, n, edges):
#         children = collections.defaultdict(set)
#         for s, t in edges:
#             children[s].add(t)
#             children[t].add(s)
#         vertices = set(children.keys())
#         while len(vertices) > 2:
#             leaves = [x for x in children if len(children[x]) == 1]
#             for x in leaves:
#                 for y in children[x]:
#                     children[y].remove(x)
#                 del children[x]
#                 vertices.remove(x)
#         return list(vertices) if n != 1 else [0]

# # Sparse Matrix Multiplication
# # Given two sparse matrices A and B, return the result of AB.
# def multiply(self, A, B):
#     result = [[0]*len(B[0]) for _ in xrange(len(A))]
#     Bi = [[i for i,v in enumerate(l) if v] if any(l) else [] for l in B]
#     for i in xrange(len(A)):
#         if not any(A[i]):
#             continue
#         for j in xrange(len(A[0])):
#             if A[i][j]:
#                 for k in Bi[j]:
#                     result[i][k]+=A[i][j]*B[j][k]
#     return result

# # Burst Balloons
# # Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums. 
# # You are asked to burst all the balloons. If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins. 
# # Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.
# # Find the maximum coins you can collect by bursting the balloons wisely.
# def maxCoins(self, nums): # bottom up with rescursive
#     nums = [1] + nums + [1]
#     n = len(nums)
#     dp = [[0]*n for _ in xrange(n)]
#     def helper(left, right):
#         if right-left <=1: # if there are 1 or 2 elements
#             return 0
#         if dp[left][right]:
#             return dp[left][right]
#         res = []
#         for last in range(left+1, right): # choose which element to finish lastly and caculate sub array
#             res.append(nums[left]*nums[last]*nums[right]+helper(left, last)+helper(last,right))
#         dp[left][right] = max(res)
#         return dp[left][right]
#     return helper(0,len(nums)-1)

# # Super Ugly Number
# # Write a program to find the nth super ugly number.
# # Super ugly numbers are positive numbers whose all prime factors are in the given prime list primes of size k. 
# # For example, [1, 2, 4, 7, 8, 13, 14, 16, 19, 26, 28, 32] is the sequence of the first 12 super ugly numbers given primes = [2, 7, 13, 19] of size 4.
# class Solution(object):
#     def nthSuperUglyNumber(self, n, primes):
#         uglies = [1]
#         def gen(prime):
#             for ugly in uglies:
#                 yield ugly * prime
#         merged = heapq.merge(*map(gen, primes))
#         while len(uglies) < n:
#             ugly = next(merged)
#             if ugly != uglies[-1]:
#                 uglies.append(ugly)
#         return uglies[-1]

# # Binary Tree Vertical Order Traversal
# # Given a binary tree, return the vertical order traversal of its nodes' values. (ie, from top to bottom, column by column).
# # If two nodes are in the same row and column, the order should be from left to right.
# class Solution(object):
# def verticalOrder(self, root):
#     if not root: return []
#     d = {}
#     queue = [(root, 0)]
#     # index denotes which column the node is in
#     while queue:
#         curNode, index = queue.pop(0)
#         d[index] = d.get(index, []) + [curNode.val]
#         if curNode.left: queue.append((curNode.left, index-1))
#         if curNode.right: queue.append((curNode.right, index+1))
#     minIndex = min(d.keys())
#     maxIndex = max(d.keys())
#     res = []
#     for i in range(minIndex, maxIndex+1):
#         res.append(d[i])
#     return res

# # Count of Smaller Numbers After Self
# # You are given an integer array nums and you have to return a new counts array. 
# # The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].
# class Solution(object):
#     def countSmaller(self, nums):
#         rank, N, res = {val: i + 1 for i, val in enumerate(sorted(nums))}, len(nums), []
#         BITree = [0] * (N + 1)
#         def update(i):
#             while i <= N:
#                 BITree[i] += 1
#                 i += (i & -i)
#         def getSum(i):
#             s = 0
#             while i:
#                 s += BITree[i]
#                 i -= (i & -i)
#             return s
#         for x in reversed(nums):
#             res += getSum(rank[x] - 1),
#             update(rank[x])
#         return res[::-1]

# # Remove Duplicate Letters
# # Given a string which contains only lowercase letters, remove duplicate letters so that every letter appear once and only once. 
# # You must make sure your result is the smallest in lexicographical order among all possible results.
# class Solution(object):
#     def removeDuplicateLetters(self, s):
#         import collections
#         ans = ''
#         for x in range(len(set(s))):
#             top, idx = s[0], 0
#             counter = collections.Counter(s)
#             for y in range(len(s)):
#                 if top > s[y]:
#                     top, idx = s[y], y
#                 if counter[s[y]] == 1:
#                     break
#                 counter[s[y]] -= 1
#             ans += top
#             s = s[idx+1:].replace(top,'')
#         return ans

# # Shortest Distance from All Buildings
# class Solution(object):
#     def traverse(self, startPoint, m, n, grid, distance):
#         queue, level, nth = collections.deque([startPoint]), 1, self.nth
#         while queue:
#             for _ in xrange(len(queue)):
#                 i, j = queue.popleft()
#                 for x, y in ((0, -1), (-1, 0), (0, 1), (1, 0)):
#                     ii, jj = i + x, j + y
#                     if 0 <= ii < m and 0 <= jj < n and grid[ii][jj] == nth + 1:
#                         queue.append((ii, jj))
#                         distance[ii][jj] += level
#                         grid[ii][jj] = nth
#             level += 1
#         self.nth -= 1
#     def shortestDistance(self, grid):
#         m, n, self.nth = len(grid), len(grid[0]), -1
#         distance = [[0] * n for _ in xrange(m)]
#         buildingNumber = len([
#             self.traverse((i, j), m, n, grid, distance)
#             for i, row in enumerate(grid)
#             for j, num in enumerate(row) if num == 1
#         ])
#         return min([
#             distance[i][j]
#             for i, row in enumerate(grid)
#             for j, num in enumerate(row)
#             if num == -buildingNumber
#         ] or [-1])

# # Maximum Product of Word Lengths
# # Given a string array words, find the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters. 
# # You may assume that each word will contain only lower case letters. If no such two words exist, return 0.
# class Solution(object):
#     def maxProduct(self, words):
#         nums = []
#         size = len(words)
#         for w in words:
#             nums += sum(1 << (ord(x) - ord('a')) for x in set(w)),
#         ans = 0
#         for x in range(size):
#             for y in range(size):
#                 if not (nums[x] & nums[y]):
#                     ans = max(len(words[x]) * len(words[y]), ans)
#         return ans

# # Bulb Switcher
# # There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second bulb. On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). 
# # For the ith round, you toggle every i bulb. For the nth round, you only toggle the last bulb. Find how many bulbs are on after n rounds.
# class Solution(object):
#     def bulbSwitch(self, n):
#         return int(math.sqrt(n))

# # Generalized Abbreviation
# # Write a function to generate the generalized abbreviations of a word.
# # Example:
# # Given word = "word", return the following list (order does not matter):
# # ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]
# def abbreviate(s):
#     if not s:
#         return ['']
#     else:
#         res = abbreviate(s[:-1])
#         return [w + s[-1] for w in res] + [extendWordWith1(w) for w in res]
# def extendWordWith1(word):
#     if word and word[-1].isdigit():
#         idx = len(word)-1
#         while idx-1>=0 and word[idx-1].isdigit():
#             idx -=1
#         return word[:idx] + str(int(word[idx:]) + 1)
#     else:
#         return word+'1'

# # Create Maximum Number
# # Given two arrays of length m and n with digits 0-9 representing two numbers. Create the maximum number of length k <= m + n from digits of the two. 
# # The relative order of the digits from the same array must be preserved. Return an array of the k digits. You should try to optimize your time and space complexity.
# class Solution(object):
#     def maxNumber(self, nums1, nums2, k):
#         def getMax(nums, t):
#             ans = []
#             size = len(nums)
#             for x in range(size):
#                 while ans and len(ans) + size - x > t and ans[-1] < nums[x]:
#                     ans.pop()
#                 if len(ans) < t:
#                     ans += nums[x],
#             return ans
#         def merge(nums1, nums2):
#             ans = []
#             while nums1 or nums2:
#                 if nums1 > nums2:
#                     ans += nums1[0],
#                     nums1 = nums1[1:]
#                 else:
#                     ans += nums2[0],
#                     nums2 = nums2[1:]
#             return ans
#         len1, len2 = len(nums1), len(nums2)
#         res = []
#         for x in range(max(0, k - len2), min(k, len1) + 1):
#             tmp = merge(getMax(nums1, x), getMax(nums2, k - x))
#             res = max(tmp, res)
#         return res

# # Coin Change
# # You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
# class Solution(object):
#     def coinChange(self, coins, amount):
#         dp = [0] + [-1] * amount
#         for x in range(amount):
#             if dp[x] < 0:
#                 continue
#             for c in coins:
#                 if x + c > amount:
#                     continue
#                 if dp[x + c] < 0 or dp[x + c] > dp[x] + 1:
#                     dp[x + c] = dp[x] + 1
#         return dp[amount]

# # Number of Connected Components in an Undirected Graph
# # Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to find the number of connected components in an undirected graph.
# import collections
# class Solution(object):
#     def depthFirstSearch(self, adj_list, visited, vertex):
#         visited[vertex] = True
#         for neighbor in adj_list[vertex]:
#             if not visited[neighbor]:
#                 self.depthFirstSearch(adj_list, visited, neighbor)
#     def countComponents(self, n, edges):
#         adj_list = collections.defaultdict(list)
#         for edge in edges:
#             adj_list[edge[0]].append(edge[1])
#             adj_list[edge[1]].append(edge[0])
#         visited = [False for i in xrange(n)]
#         connected_components = 0
#         for vertex in xrange(n):
#             if not visited[vertex]:
#                 self.depthFirstSearch(adj_list, visited, vertex)
#                 connected_components += 1
#         return connected_components

# # Wiggle Sort II
# # Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]....
# class Solution(object):
#     def wiggleSort(self, nums):
#         size = len(nums)
#         snums = sorted(nums)
#         for x in range(1, size, 2) + range(0, size, 2):
#             nums[x] = snums.pop()

# # Maximum Size Subarray Sum Equals k
# # Given an array nums and a target value k, find the maximum length of a subarray that sums to k. If there isn't one, return 0 instead.
# def maxSubArrayLen(self, nums, k):
#     ans, acc = 0, 0
#     mp = {0:-1}
#     for i in xrange(len(nums)):
#         acc += nums[i]
#         if acc not in mp:
#             mp[acc] = i 
#         if acc-k in mp:
#             ans = max(ans, i-mp[acc-k])
#     return ans

# # Power of Three
# # Given an integer, write a function to determine if it is a power of three.
# class Solution(object):
#     def isPowerOfThree(self, n):
#         return n > 0 and 1162261467 % n == 0

# # Count of Range Sum
# # Given an integer array nums, return the number of range sums that lie in [lower, upper] inclusive.
# # Range sum S(i, j) is defined as the sum of the elements in nums between indices i and j (i ≤ j), inclusive.
# class Solution(object):
#     def countRangeSum(self, nums, lower, upper):
#         sums = nums[:]
#         for x in range(1, len(sums)):
#             sums[x] += sums[x - 1]
#         osums = sorted(set(sums))
#         ft = FenwickTree(len(osums))
#         ans = 0
#         for sumi in sums:
#             left = bisect.bisect_left(osums, sumi - upper)
#             right = bisect.bisect_right(osums, sumi - lower)
#             ans += ft.sum(right) - ft.sum(left) + (lower <= sumi <= upper)
#             ft.add(bisect.bisect_right(osums, sumi), 1)
#         return ans
# class FenwickTree(object):
#     def __init__(self, n):
#         self.n = n
#         self.sums = [0] * (n + 1)
#     def add(self, x, val):
#         while x <= self.n:
#             self.sums[x] += val
#             x += self.lowbit(x)
#     def lowbit(self, x):
#         return x & -x
#     def sum(self, x):
#         res = 0
#         while x > 0:
#             res += self.sums[x]
#             x -= self.lowbit(x)
#         return res

# # Odd Even Linked List
# # Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.
# class Solution(object):
#     def oddEvenList(self, head):
#         if head is None: return head
#         odd = oddHead = head
#         even = evenHead = head.next
#         while even and even.next:
#             odd.next = even.next
#             odd = odd.next
#             even.next = odd.next
#             even = even.next
#         odd.next = evenHead
#         return oddHead

# # Longest Increasing Path in a Matrix
# # Given an integer matrix, find the length of the longest increasing path.
# # From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).
# def longestIncreasingPath(self, matrix):
#     matrix = {i + j*1j: val
#               for i, row in enumerate(matrix)
#               for j, val in enumerate(row)}
#     length = {}
#     for z in sorted(matrix, key=matrix.get):
#         length[z] = 1 + max([length[Z]
#                              for Z in z+1, z-1, z+1j, z-1j
#                              if Z in matrix and matrix[z] > matrix[Z]]
#                             or [0])
#     return max(length.values() or [0])

# # Patching Array
# # Given a sorted positive integer array nums and an integer n, add/patch elements to the array such that any number in range [1, n] inclusive can be formed by the sum of some elements in the array. Return the minimum number of patches required.
# class Solution(object):
#     def minPatches(self, nums, n):
#         idx, total, ans = 0, 1, 0
#         size = len(nums)
#         while total <= n:
#             if idx < size and nums[idx] <= total:
#                 total += nums[idx]
#                 idx += 1
#             else:
#                 total <<= 1
#                 ans += 1
#         return ans

# # Verify Preorder Serialization of a Binary Tree
# # Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of a binary tree. Find an algorithm without reconstructing the tree.
# class Solution(object):
#     def isValidSerialization(self, preorder):
#         stack = collections.deque()
#         for item in preorder.split(','):
#             stack.append(item)
#             while len(stack) >= 3 and \
#                   stack[-1] == stack[-2] == '#' and \
#                   stack[-3] != '#':
#                 stack.pop(), stack.pop(), stack.pop()
#                 stack.append('#')
#         return len(stack) == 1 and stack[0] == '#'

# # Reconstruct Itinerary
# # Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.
# class Solution(object):
#     def findItinerary(self, tickets):
#         routes = collections.defaultdict(list)
#         for s, e in tickets:
#             routes[s].append(e)
#         def solve(start):
#             left, right = [], []
#             for end in sorted(routes[start]):
#                 if end not in routes[start]:
#                     continue
#                 routes[start].remove(end)
#                 subroutes = solve(end)
#                 if start in subroutes:
#                     left += subroutes
#                 else:
#                     right += subroutes
#             return [start] + left + right
#         return solve("JFK")

# # Largest BST Subtree
# # Given a binary tree, find the largest subtree which is a Binary Search Tree (BST), where largest means subtree with largest number of nodes in it.
# class SubTree(object):
#     def __init__(self, largest, n, min, max):
#         self.largest = largest  # largest BST
#         self.n = n              # number of nodes in this ST
#         self.min = min          # min val in this ST
#         self.max = max          # max val in this ST
# class Solution(object):
#     def largestBSTSubtree(self, root):
#         res = self.dfs(root)
#         return res.largest
#     def dfs(self, root):
#         if not root:
#             return SubTree(0, 0, float('inf'), float('-inf'))
#         left = self.dfs(root.left)
#         right = self.dfs(root.right)
#         if root.val > left.max and root.val < right.min:  # valid BST
#             n = left.n + right.n + 1
#         else:
#             n = float('-inf')
#         largest = max(left.largest, right.largest, n)
#         return SubTree(largest, n, min(left.min, root.val), max(right.max, root.val))

# Increasing Triplet Subsequence
# Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.
# class Solution(object):
#     def increasingTriplet(self, nums):
#         a = b = None
#         for n in nums:
#             if a is None or a >= n:
#                 a = n
#             elif b is None or b >= n:
#                 b = n
#             else:
#                 return True
#         return False

# # Self Crossing
# # You are given an array x of n positive numbers. You start at point (0,0) and moves x[0] metres to the north, then x[1] metres to the west, x[2] metres to the south, x[3] metres to the east and so on. In other words, after each move your direction changes counter-clockwise.
# class Solution(object):
#     def isSelfCrossing(self, x):
#         n = len(x)
#         if n < 4: return False
#         t1, (t2, t3, t4) = 0, x[:3]
#         increase = True if t2 < t4 else False
#         for i in xrange(3, n):
#             t5 = x[i]
#             if increase and t3 >= t5:
#                 if t5 + t1 - t3 < 0 or i + 1 < n and x[i + 1] + t2 - t4 < 0:
#                     increase = False
#                 elif i + 1 < n:
#                     return True
#             elif not increase and t3 <= t5:
#                 return True
#             t1, t2, t3, t4 = t2, t3, t4, t5
#         return False

# # Palindrome Pairs
# # Given a list of unique words. Find all pairs of distinct indices (i, j) in the given list, so that the concatenation of the two words, i.e. words[i] + words[j] is a palindrome.
# class Solution(object):
#     def palindromePairs(self, words):
#         wmap = {y : x for x, y in enumerate(words)}
#         def isPalindrome(word):
#             size = len(word)
#             for x in range(size / 2):
#                 if word[x] != word[size - x - 1]:
#                     return False
#             return True
#         ans = set()
#         for idx, word in enumerate(words):
#             if "" in wmap and word != "" and isPalindrome(word):
#                 bidx = wmap[""]
#                 ans.add((bidx, idx))
#                 ans.add((idx, bidx))
#             rword = word[::-1]
#             if rword in wmap:
#                 ridx = wmap[rword]
#                 if idx != ridx:
#                     ans.add((idx, ridx))
#                     ans.add((ridx, idx))
#             for x in range(1, len(word)):
#                 left, right = word[:x], word[x:]
#                 rleft, rright = left[::-1], right[::-1]
#                 if isPalindrome(left) and rright in wmap:
#                     ans.add((wmap[rright], idx))
#                 if isPalindrome(right) and rleft in wmap:
#                     ans.add((idx, wmap[rleft]))
#         return list(ans)

# # House Robber III
# # The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root." Besides the root, each house has one and only one parent house. 
# # After a tour, the smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses were broken into on the same night.
# class Solution(object):
#     def rob(self, root):
#         valMap = dict()
#         def solve(root, path):
#             if root is None: return 0
#             if path not in valMap:
#                 left, right = root.left, root.right
#                 ll = lr = rl = rr = None
#                 if left:  ll, lr = left.left, left.right
#                 if right: rl, rr = right.left, right.right
#                 passup = solve(left, path + 'l') + solve(right, path + 'r')
#                 grabit = root.val + solve(ll, path + 'll') + solve(lr, path + 'lr') \
#                          + solve(rl, path + 'rl') + solve(rr, path + 'rr')
#                 valMap[path] = max(passup, grabit)
#             return valMap[path]
#         return solve(root, '')

# # Counting Bits
# # Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.
# class Solution(object):
#     def countBits(self, num):
#         ans = [0]
#         for x in range(1, num + 1):
#             ans += ans[x >> 1] + (x & 1),
#         return ans

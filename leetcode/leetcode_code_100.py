# -*- coding: utf-8 -*- 

# # Two Sum
# # Given an array of integers, find two numbers such that they add up to a specific target number.
# class Solution:
#     def twoSum(self, nums, target):
#         lookup = {}
#         for i, num in enumerate(nums):
#             if target - num in lookup:
#                 return (lookup[target - num] + 1, i + 1)
#             lookup[num] = i

# # Add Two Numbers
# # You are given two linked lists representing two non-negative numbers. 
# # The digits are stored in reverse order and each of their nodes contain a single digit.
# # Add the two numbers and return it as a linked list.
# class ListNode:
# 	def __init__(self, x):
# 		self.val = x
# 		self.next = None
# class Solution:
# 	def addTwoNumbers(self, l1, l2):
# 		Carry=0; head=ListNode(0); curr=head
# 		while l1 and l2:
# 			Sum, Carry = (l1.val+l2.val+Carry)%10, (l1.val+l2.val+Carry)/10
# 			curr.next = ListNode(Sum)
# 			curr, l1, l2 = curr.next, l1.next, l2.next
# 		while l1:
# 			Sum, Carry = (l1.val+Carry)%10, (l1.val+Carry)/10
# 			curr.next = ListNode(Sum)
# 			curr, l1 = curr.next, l1.next
# 		while l2:
# 			Sum, Carry = (l2.val+Carry)%10, (l2.val+Carry)/10
# 			curr.next = ListNode(Sum)
# 			curr, l2 = curr.next, l2.next
# 		if Carry!=0:
# 			curr.next = ListNode(Carry)
# 			curr = curr.next
# 		return head.next
# if __name__ == '__main__':
#     a, a.next, a.next.next = ListNode(2), ListNode(4), ListNode(3)
#     b, b.next, b.next.next = ListNode(5), ListNode(6), ListNode(4)
#     result = Solution().addTwoNumbers(a, b)
#     print "{0} -> {1} -> {2}".format(result.val, result.next.val, result.next.next.val)

# # Longest Substring Without Repeating Characters
# # Given a string, find the length of the longest substring without repeating characters.
# class Solution:
# 	def lengthOfLongestSubstring(self, s):
# 		start, longest, visited = 0, 0, {}
# 		for i, char in enumerate(s):
# 			if char in visited:
# 				while s[start]!=char:
# 					del visited[s[start]]
# 					start += 1
# 				start += 1
# 			else:
# 				visited[char] = True
# 			longest = max(longest, len(visited.keys()))
# 		return longest

# # Median of Two Sorted Arrays
# # There are two sorted arrays nums1 and nums2 of size m and n respectively. Find the median of the two sorted arrays.
# class Solution:
#     def findK(self, A, B, k):
#         if len(A)>len(B): return self.findK(A,B,k)
#         if len(A) == 0: return B[k-1]
#         if k == 1: return min(A[0],B[0])
#         pa = min(k/2,len(A)); pb = k-pa
#         return self.findK(A[pa:],B,k-pa) if A[pa-1]<=B[pb-1] else self.findK(A,B[pb:],k-pb)
#     def findMedianSortedArrays(self, nums1, nums2):
#         if (len(nums1)+len(nums2))%2 == 1:
#             return self.findK(nums1,nums2,(len(nums1)+len(nums2))/2+1)
#         else:
#             return 0.5*(self.findK(nums1,nums2,(len(nums1)+len(nums2))/2)+self.findK(nums1,nums2,(len(nums1)+len(nums2))/2+1))

# # Longest Palindromic Substring 
# # Given a string S, find the longest palindromic substring in S.
# class Solution:
# 	def longestPalindrome(self, s):
# 		if len(s)==1: return s
# 		p=s[0]
# 		for i in range(1,len(s)):
# 			r1, r2 = self.palin(s,i,i), self.palin(s,i-1,i)
# 			if len(r1)>len(p): p=r1
# 			if len(r2)>len(p): p=r2
# 		return p
# 	def palin(self,s,begin,end):
# 		while begin>=0 and end<=len(s)-1 and s[begin]==s[end]:
# 			begin-=1; end+=1
# 		return s[begin+1:end]
# class Solution:
#     def longestPalindrome(self, s):
#         start, maxlen, d = 0, 1, [[False for j in xrange(len(s))] for i in xrange(len(s))]
#         for i in xrange(len(s)):
#             d[i][i] = True
#             for j in xrange(i):
#                 d[j][i] = s[j]==s[i] and (i-j<=1 or d[j+1][i-1])
#                 if d[j][i] and i-j+1>maxlen:
#                     start, maxlen = j, i-j+1
#         return s[start:start+maxlen]

# # ZigZag Conversion 
# # The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this
# class Solution:
#     # @return a string
#     def convert(self, s, numRows):
# 		t, index, step = ["" for i in xrange(numRows)], -1, 1
# 		if numRows == 1: return s
# 		for i in xrange(len(s)):
# 			index += step
# 			if index==numRows:
# 				index-=2; step=-1
# 			elif index==-1:
# 				index+=2; step=+1
# 			t[index]+=s[i]
# 		return ''.join(t)
# print Solution().convert("PAYPALISHIRING", 1)

# # Reverse Integer
# # Reverse digits of an integer.
# class Solution:
# 	def reverse(self, x):
# 		a = 0
# 		b = x if x > 0 else -x
# 		while b:
# 			if a > (2**31-1) / 10:
# 				return 0
# 			else:
# 				a = a * 10 + b % 10
# 				b = b / 10
# 		return a if x > 0 else -a

# # String to Integer (atoi) 
# # Implement atoi to convert a string to an integer.
# class Solution:
#     def myAtoi(self, str):
#         INT_MAX, INT_MIN, result, sign = 2147483647, -2147483648, 0, 1
#         str, index = str.strip(), 0
#         if len(str)==0: return 0
#         if str[0]=="-": sign=-1
#         if str[0]in["-","+"]: index+=1
#         while index <= len(str)-1 and str[index].isdigit():
#             result = result*10+(ord(str[index])-ord("0"))*sign
#             if sign==1 and result>=INT_MAX:
#                 return INT_MAX
#             elif sign==-1 and result<=INT_MIN:
#                 return INT_MIN
#             index+=1
#         return result
# print Solution().myAtoi("+1")

# # Palindrome Number
# # Determine whether an integer is a palindrome. Do this without extra space.
# class Solution:
#     def isPalindrome(self, x):
#         if x < 0:
#             return False
#         copy, reverse = x, 0
#         while copy:
#             reverse *= 10
#             reverse += copy % 10
#             copy /= 10
#         return x == reverse

# # Regular Expression Matching
# # '.' Matches any single character.
# # '*' Matches zero or more of the preceding element.
# # isMatch("aa","aa") → true
# # isMatch("aa", "a*") → true
# # isMatch("aa", ".*") → true
# # isMatch("ab", ".*") → true
# # isMatch("aab", "c*a*b") → true
# class Solution:
# 	def isMatch(self, s, p):
# 		result = [[False for j in xrange(len(p)+1)] for i in xrange(len(s)+1)]
# 		result[0][0] = True
# 		for i in xrange(2, len(p)+1): #考虑s为空字符串的情况
# 			if p[i-1] == '*':
# 				result[0][i] = result[0][i-2]
# 		for i in xrange(1,len(s)+1):
# 			for j in xrange(1,len(p)+1):
# 				if p[j-1] == '*':
# 					result[i][j] = result[i][j-2] or (result[i-1][j] and (s[i-1] == p[j-2] or p[j-2]=='.'))
# 				else:
# 					result[i][j] = result[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '.')
# 		return result[len(s)][len(p)]
# print Solution().isMatch("ab", ".*")

# # Container With Most Water
# # Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). 
# # n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). 
# # Find two lines, which together with x-axis forms a container, such that the container contains the most water.
# class Solution:
#     def maxArea(self, height):
#         start, end, area = 0, len(height)-1, 0
#         while start != end:
#             area = max(area,(end-start)*min(height[start],height[end]))
#             if height[start]<height[end]:
#                 start += 1
#             else:
#                 end -=1
#         return area
  
# # Integer to Roman      
# class Solution:
# 	def intToRoman(self, num):
# 		numeral_map = {1: "I", 4: "IV", 5: "V", 9: "IX", 10: "X", 40: "XL", 50: "L", 90: "XC", 100: "C", 400: "CD", 500: "D", 900: "CM", 1000: "M"}
# 		keyset, result = sorted(numeral_map.keys()), ""
# 		while num > 0:
# 			for key in reversed(keyset):
# 				quo = num/key
# 				num -= key*quo
# 				result += numeral_map[key]*quo
# 		return result

# # Roman to Integer
# class Solution:
#     def romanToInt(self, s):
#         numeral_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C":100, "D": 500, "M": 1000}
#         decimal = 0
#         for i in xrange(len(s)):
#             if i >= 1 and numeral_map[s[i]]>numeral_map[s[i-1]]:
#                 decimal += numeral_map[s[i]]-2*numeral_map[s[i-1]]
#             else:
#                 decimal += numeral_map[s[i]]
#         return decimal

# # Longest Common Prefix
# # Write a function to find the longest common prefix string amongst an array of strings.
# class Solution:
# 	def longestCommonPrefix(self, strs):
# 		if len(strs) == 0:
# 			return ""
# 		i = 0
# 		while i < min([len(str) for str in strs]) and all([str[i]==strs[0][i] for str in strs]):
# 			i+=1
# 		return strs[0][0:i]
# print Solution().longestCommonPrefix([""])

# # 3Sum
# # Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.
# class Solution:
#     def threeSum(self, nums):
#         nums, result, i = sorted(nums), [], 0
#         while i<=len(nums) - 3:
#             j, k = i+1, len(nums)-1
#             while j < k:
#                 if nums[i]+nums[j]+nums[k] < 0:
#                     j += 1
#                 elif nums[i]+nums[j]+nums[k] > 0:
#                     k -= 1
#                 else:
#                     result.append([nums[i],nums[j],nums[k]])
#                     j, k = j+1, k-1
#                     while j < k and nums[j] == nums[j-1]: #过滤重复结果
#                         j += 1
#                     while j < k and nums[k] == nums[k+1]:
#                         k -= 1
#             i += 1
#             while i <= len(nums)-3 and nums[i] == nums[i-1]:
#                 i += 1
#         return result

# # 3Sum Closest
# # Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target.
# class Solution:
#     def threeSumClosest(self, nums, target):
#         nums, result, min_diff, i = sorted(nums), float("inf"), float("inf"), 0
#         while i < len(nums)-2:
#             j, k = i+1, len(nums)-1
#             while j < k:
#                 diff = nums[i]+nums[j]+nums[k]-target
#                 if abs(diff) < min_diff:
#                     min_diff = abs(diff)
#                     result = nums[i]+nums[j]+nums[k]
#                 if diff < 0:
#                     j += 1
#                 elif diff > 0:
#                     k -= 1
#                 else:
#                     return target
#             i += 1
#             while i <= len(nums)-3 and nums[i] == nums[i-1]: #过滤重复结果
#                 i += 1
#         return result

# # Letter Combinations of a Phone Number
# # Given a digit string, return all possible letter combinations that the number could represent.
# class Solution:
# 	def letterCombinations(self, digits):
# 		if len(digits) == 0:
# 			return []
# 		lookup, result = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"], [""]
# 		for i in xrange(len(digits)):
# 			n, result = len(result), result*len(lookup[int(digits[i])])
# 			for j in xrange(len(result)):
# 				result[j] += lookup[int(digits[i])][j/n]
# 		return result

# # 4Sum
# # Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target?
# # Find all unique quadruplets in the array which gives the sum of target.
# class Solution:
#     def fourSum(self, num, target):
#         num, result, lookup = sorted(num), [], {}
#         for i in xrange(0,len(num)-1):
#             for j in xrange(i+1,len(num)):
#                 if not lookup.has_key(num[i]+num[j]):
#                     lookup[num[i]+num[j]] = []
#                 lookup[num[i]+num[j]].append([i,j])
#         for i in lookup.keys():
#             if lookup.has_key(target-i):
#                 for x in lookup[i]:
#                     for y in lookup[target-i]:
#                         [a, b], [c, d] = x, y
#                         if a!=c and a!=d and b!=c and b!=d:
#                             quad = sorted([num[a], num[b], num[c], num[d]])
#                             if quad not in result:
#                                 result.append(quad)
#         return result

# # Remove Nth Node From End of List
# # Given a linked list, remove the nth node from the end of list and return its head.
# class ListNode:
# 	def __init__(self, x):
# 		self.val = x
# 		self.next = None
# class Solution:
# 	def removeNthFromEnd(self, head, n):
# 		dummy = ListNode(-1)
# 		dummy.next = head
# 		slow, fast = dummy, dummy
# 		for i in xrange(n):
# 			fast = fast.next
# 		while fast.next:
# 			slow, fast = slow.next, fast.next
# 		slow.next = slow.next.next
# 		return dummy.next

# # Valid Parentheses
# # Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
# class Solution:
#     def isValid(self, s):
#         stack, lookup = [], {"(": ")", "{": "}", "[": "]"}
#         for parenthese in s:
#             if parenthese in lookup:
#                 stack.append(parenthese)
#             elif len(stack) == 0 or lookup[stack.pop()] != parenthese:
#                 return False
#         return len(stack) == 0

# # Merge Two Sorted Lists
# class Solution:
# 	def mergeTwoLists(self, l1, l2):
# 		dummy = ListNode(0)
# 		current = dummy
# 		while l1 and l2:
# 			if l1.val < l2.val:
# 				current.next = l1
# 				l1 = l1.next
# 			else:
# 				current.next = l2
# 				l2 = l2.next
# 			current = current.next
# 		if l1:
# 			current.next = l1
# 		else:
# 			current.next = l2
# 		return dummy.next

# # Generate Parentheses
# # Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
# class Solution:
#     def generateParenthesis(self, n):
#         result  = []
#         self.generateParenthesisRecu(result,"",n,n)
#         return result
#     def generateParenthesisRecu(self, result, current, left, right):
#         if left==0 and right==0:
#             result.append(current)
#         if left>0:
#             self.generateParenthesisRecu(result,current+"(",left-1,right)
#         if left<right:
#             self.generateParenthesisRecu(result,current+")",left,right-1)

# # Merge k Sorted Lists
# # Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
# class Solution:
#     def mergeKLists(self, lists):
#         import heapq
#         heap = []
#         for node in lists:
#             if node != None: heap.append((node.val, node))
#         heapq.heapify(heap)
#         head = ListNode(0); curr = head
#         while heap:
#             pop = heapq.heappop(heap)
#             curr.next = pop[1]
#             curr = curr.next
#             if pop[1].next:
#                 heapq.heappush(heap, (pop[1].next.val, pop[1].next))
#         return head.next

# # Swap Nodes in Pairs
# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class Solution:
#     def swapPairs(self, head):
#         dummy = ListNode(0)
#         dummy.next = head
#         current = dummy
#         while current.next and current.next.next:
#             next_one, next_two, next_three = current.next, current.next.next, current.next.next.next
#             current.next = next_two
#             next_two.next = next_one
#             next_one.next = next_three
#             current = next_one
#         return dummy.next

# # Reverse Nodes in k-Group
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class Solution:
#     def reverseKGroup(self, head, k):
#         dummy = ListNode(-1)
#         dummy.next = head
#         cur, cur_dummy, length = head, dummy, 0
#         while cur:
#             cur_next = cur.next
#             length = (length+1) % k
#             if length == 0:
#                 dummy_next = cur_dummy.next
#                 self.reverse(cur_dummy, cur.next)
#                 cur_dummy = dummy_next
#             cur = cur_next
#         return dummy.next
#     def reverse(self, begin, end):
#             first = begin.next
#             cur = first.next
#             while cur != end:
#                 first.next = cur.next
#                 cur.next = begin.next
#                 begin.next = cur
#                 cur = first.next

# # Remove Duplicates from Sorted Array
# class Solution:
#     def removeDuplicates(self, A):
#         if len(A) == 0:
#             return 0
#         tail = 0
#         for head in xrange(1,len(A)):
#             if A[head] != A[head-1]:
#                 tail+=1
#                 A[tail] = A[head]
#         return tail+1

# # Remove Element
# # Given an array and a value, remove all instances of that value in place and return the new length.
# class Solution:
#     def removeElement(self, A, elem):
#         if len(A) == 0:
#             return 0
#         tail = -1
#         for i in xrange(len(A)):
#             if A[i] != elem:
#                tail += 1
#                A[tail] = A[i]
#         return tail+1

# Implement strStr()
# class Solution:
# 	def strStr(self, haystack, needle):
# 		for i in xrange(len(haystack)-len(needle)+1):
# 			if haystack[i:i+len(needle)]==needle:
# 				return i
# 		return -1

# # Divide Two Integers
# # Divide two integers without using multiplication, division and mod operator.
# class Solution:
# 	def divide(self, dividend, divisor):
# 		sign = 1 if (dividend > 0 and divisor > 0) or (dividend < 0 and divisor < 0) else -1
# 		dividend, divisor, quotient = abs(dividend), abs(divisor), 0
# 		while dividend >= divisor:
# 			k = 0
# 			divisor_curr = divisor
# 			while dividend >= divisor_curr:
# 				dividend -= divisor_curr #被减数
# 				divisor_curr = divisor_curr<<1 #减数翻倍
# 				quotient = quotient+(1<<k) #计算商
# 				k += 1
# 		quotient = quotient * sign
# 		if int(quotient) > 2147483647: return 2147483647
# 		elif int(quotient) < -2147483648: return 2147483648
# 		else: return int(quotient)

# # Substring with Concatenation of All Words
# # Find all starting indices of substring(s) in s that is a concatenation of each word in words exactly once and without any intervening characters.
# class Solution:
# 	def findSubstring(self, s, words):
# 		result = []
# 		for start in xrange(len(s)-len(words[0])*len(words)+1):
# 			lists = [s[i:i+len(words[0])] for i in xrange(start,start+len(words[0])*len(words),len(words[0]))]
# 			found = True
# 			for word in words:
# 				if word in lists:
# 					lists.remove(word)
# 				else:
# 					found = False
# 					break
# 			if found:
# 				result.append(start)
# 		return result

# # Next Permutation
# # 6,8,7,4,3,2 -> 7,2,3,4,6,8
# class Solution:
#     def nextPermutation(self, num):
#         partitionIndex = -1
#         for i in reversed(xrange(len(num)-1)):
#             if num[i] < num[i+1]:
#                 partitionIndex = i
#                 break
#         if partitionIndex != -1:
#             for i in reversed(xrange(len(num))):
#                 if num[partitionIndex] < num[i]:
#                     num[i], num[partitionIndex] = num[partitionIndex], num[i]
#                     break
#         i, j = partitionIndex+1, len(num)-1
#         while i < j:
#             num[i], num[j] = num[j], num[i]
#             i += 1
#             j -= 1

# # Longest Valid Parentheses
# # ")()())" -> "()()"
# class Solution:
#     def longestValidParentheses(self, s):
#         stack, maxLen = [(-1,')')], 0
#         for i in xrange(len(s)):
#             if s[i] == ')' and stack[-1][1] == '(':
#                 stack.pop()
#                 maxLen = max(maxLen,i-stack[-1][0])
#             else:
#                 stack.append((i,s[i]))
#         return maxLen

# # Search in Rotated Sorted Array 
# # 0,1,2,4,5,6,7 -> 4,5,6,7,0,1,2
# class Solution:
#     def search(self, nums, target):
#         left, right = 0, len(A)-1
#         while left <= right:
#             mid = (left+right)/2
#             if A[mid] == target: return mid
#             if A[left] <= A[mid]:
#                 if A[left] <= target < A[mid]: right = mid-1
#                 else: left = mid+1
#             else:
#                 if A[mid] < target <= A[right]: left = mid+1
#                 else: right = mid-1
#         return -1

# # Search for a Range
# # Given [5, 7, 7, 8, 8, 10] and target value 8, return [3, 4]
# class Solution:
#     def searchRange(self, A, target):
#         begin, end = -1, -1
#         back, front = 0, len(A)-1
#         while back <= front:
#             mid = (front+back)/2
#             if A[mid] > target:
#                 front = mid-1
#             elif A[mid] < target:
#                 back = mid+1
#             else:
#                 front = mid-1
#                 begin = mid
#         back, front = 0, len(A)-1
#         while back <= front:
#             mid = (front+back)/2
#             if A[mid] > target:
#                 front = mid-1
#             elif A[mid] < target:
#                 back = mid+1
#             else:
#                 back = mid+1
#                 end = mid
#         return [begin, end]

# # Search Insert Position
# # Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
# class Solution:
#     def searchInsert(self, nums, target):
#         low, high = 0, len(nums)-1
#         while low <= high:
#             mid = (high+low)/2
#             if nums[mid] == target:
#                 return mid
#             elif nums[mid] < target:
#                 low = mid+1
#             else:
#                 high = mid-1
#         return low

# # Valid Sudoku
# class Solution:
#     def isValidSudoku(self, board):
#         for i in xrange(9):
#             if not self.isValidList([board[i][j] for j in xrange(9)]) or not self.isValidList([board[j][i] for j in xrange(9)]):
#                 return False
#         for i in xrange(3):
#             for j in xrange(3):
#                 if not self.isValidList([board[m][n] for n in xrange(3*j,3*j+3) for m in xrange(3*i,3*i+3)]):
#                     return False
#         return True
#     def isValidList(self, xs):
#         xs = filter(lambda x: x != '.', xs)
#         return len(set(xs)) == len(xs)

# # Sudoku Solver
# class Solution:
#     def solveSudoku(self, board):
#         def isValid(x,y):
#             val=board[x][y]; board[x][y]='X'
#             for i in range(9):
#                 if board[i][y]==val: return False
#             for i in range(9):
#                 if board[x][i]==val: return False
#             for i in range(3):
#                 for j in range(3):
#                     if board[(x/3)*3+i][(y/3)*3+j]==val: return False
#             board[x][y]=val
#             return True
#         def dfs():
#             for i in range(9):
#                 for j in range(9):
#                     if board[i][j]=='.':
#                         for k in '123456789':
#                             board[i][j]=k
#                             if isValid(i,j) and dfs():
#                                 return True
#                             board[i][j]='.'
#                         return False
#             return True
#         dfs()

# # Count and Say
# # 1, 11, 21, 1211, 111221, ...
# class Solution:
#     def countAndSay(self, n):
#         seq = "1"
#         for i in xrange(n - 1):
#             seq = self.getNext(seq)
#         return seq
#     def getNext(self, seq):
#         i, next_seq = 0, ""
#         while i <= len(seq)-1:
#             cnt = 1        
#             while i <= len(seq)-2 and seq[i] == seq[i+1]:
#                 cnt += 1
#                 i += 1
#             next_seq += str(cnt)+seq[i]
#             i += 1
#         return next_seq

# # Combination Sum
# # Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
# # The same repeated number may be chosen from C unlimited number of times.
# # All numbers (including target) will be positive integers.
# # Elements in a combination (a1, a2, … , ak) must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak).
# # The solution set must not contain duplicate combinations.
# class Solution:
#     def combinationSum(self, candidates, target):
#         result = []
#         self.combinationSumRecu(sorted(candidates), result, 0, [], target)
#         return result
#     def combinationSumRecu(self, candidates, result, start, intermediate, target):
#         if target == 0:
#             result.append(intermediate)
#         while start < len(candidates) and candidates[start] <= target:
#             self.combinationSumRecu(candidates, result, start, intermediate + [candidates[start]], target - candidates[start])
#             start += 1

# # Combination Sum II
# # Each number in C may only be used once in the combination.
# class Solution:
#     def combinationSum2(self, candidates, target):
#         result = []
#         self.combinationSumRecu(sorted(candidates), result, 0, [], target)
#         return result
#     def combinationSumRecu(self, candidates, result, start, intermediate, target):
#         if target == 0:
#             result.append(intermediate)
#         prev = 0
#         while start < len(candidates) and candidates[start] <= target:
#             if prev != candidates[start]:
#                 self.combinationSumRecu(candidates,result,start+1,intermediate+[candidates[start]],target-candidates[start])
#                 prev = candidates[start]
#             start += 1

# # First Missing Positive
# # Given an unsorted integer array, find the first missing positive integer.
# class Solution:
#     def firstMissingPositive(self, A):
#         i = 0
#         while i < len(A):
#             if A[i] > 0 and A[i] - 1 < len(A) and A[i] != A[A[i]-1]:
#                 A[A[i]-1], A[i] = A[i], A[A[i]-1]
#             else:
#                 i += 1
#         for i, integer in enumerate(A):
#             if integer != i + 1:
#                 return i + 1
#         return len(A) + 1

# # Trapping Rain Water
# # Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
# class Solution:
#     def trap(self, A):
#         leftmosthigh = [0 for i in range(len(A))]
#         leftmax = 0
#         for i in range(len(A)):
#             if A[i] > leftmax: leftmax = A[i]
#             leftmosthigh[i] = leftmax
#         sum = 0
#         rightmax = 0
#         for i in reversed(range(len(A))):
#             if A[i] > rightmax: rightmax = A[i]
#             if min(rightmax, leftmosthigh[i]) > A[i]:
#                 sum += min(rightmax, leftmosthigh[i]) - A[i]
#         return sum

# # Multiply Strings
# class Solution:
#     def multiply(self, num1, num2):
#         num1 = num1[::-1]; num2 = num2[::-1]
#         arr = [0 for i in range(len(num1)+len(num2))]
#         for i in range(len(num1)):
#             for j in range(len(num2)):
#                 arr[i+j] += int(num1[i]) * int(num2[j])
#         ans = []
#         for i in range(len(arr)):
#             digit = arr[i] % 10
#             carry = arr[i] / 10
#             if i < len(arr)-1:
#                 arr[i+1] += carry
#             ans.insert(0, str(digit))
#         while ans[0] == '0' and len(ans) > 1:
#             del ans[0]
#         return ''.join(ans)    

# # Wildcard Matching
# # '?' Matches any single character.
# # '*' Matches any sequence of characters (including the empty sequence).
# # isMatch("aa","aa") → true
# # isMatch("aa", "*") → true
# # isMatch("aa", "a*") → true
# # isMatch("ab", "?*") → true
# # isMatch("aab", "c*a*b") → false
# class Solution:
#     def isMatch(self, s, p):
#         p_ptr, s_ptr, last_s_ptr, last_p_ptr = 0, 0, -1, -1
#         while s_ptr < len(s):
#             if p_ptr < len(p) and (s[s_ptr] == p[p_ptr] or p[p_ptr] == '?'):
#                 s_ptr += 1
#                 p_ptr += 1
#             elif p_ptr < len(p) and p[p_ptr] == '*':
#                 p_ptr += 1
#                 last_s_ptr = s_ptr
#                 last_p_ptr = p_ptr
#             elif last_p_ptr != -1:
#                 last_s_ptr += 1
#                 s_ptr = last_s_ptr
#                 p_ptr = last_p_ptr
#             else:
#                 return False
#         while p_ptr < len(p) and p[p_ptr] == '*':
#             p_ptr += 1
#         return p_ptr == len(p)

# Jump Game II
# Given an array of non-negative integers, you are initially positioned at the first index of the array.
# Each element in the array represents your maximum jump length at that position.
# Your goal is to reach the last index in the minimum number of jumps.
# For example:
# Given array A = [2,3,1,1,4]
# The minimum number of jumps to reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps to the last index.)
# class Solution:
#     # @param A, a list of integers
#     # @return an integer
#     def jump(self, A):
#         # steps[i] means the minimum number of jumps needed to reach the position i.
#         steps = [0] * len(A)
#         for index in xrange(len(A)):
#             preSteps = steps[index] # Minimum jumps needing to reach position index
#             step = A[index]         # Maximum jump length at position index
#             # Try to jump with steps of "step" to 1
#             for jump in xrange(step, 0, -1):
#                 if index + jump >= len(steps):  continue    # Out of range
#                 if steps[index + jump] != 0:    break       # Reached before
#                 steps[index + jump] = preSteps + 1
#         return steps[-1]

# # Permutations
# # Given a collection of numbers, return all possible permutations.
# # For example,
# # [1,2,3] have the following permutations:
# # [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], and [3,2,1].
# class Solution:
#     def permute(self, num):
#         length = len(num)
#         if length == 0: return []
#         if length == 1: return [num]
#         res = []
#         for i in xrange(length):
#             for j in self.permute(num[0:i] + num[i+1:]):
#                 res.append([num[i]] + j)
#         return res

# # Permutations II
# # Given a collection of numbers that might contain duplicates, return all possible unique permutations.
# # For example,
# # [1,1,2] have the following unique permutations:
# # [1,1,2], [1,2,1], and [2,1,1].
# class Solution:
#     def permuteUnique(self, num):
#         length = len(num)
#         if length == 0: return []
#         if length == 1: return [num]
#         num.sort()
#         res = []
#         prevNum = None
#         for curr in xrange(length):
#             if num[curr] == prevNum: continue
#             prevNum = num[curr]
#             for j in self.permuteUnique(num[:curr] + num[curr + 1:]):
#                 res.append([num[curr]] + j)
#         return res

# # Rotate Image
# # You are given an n x n 2D matrix representing an image.
# # Rotate the image by 90 degrees (clockwise).
# # Follow up:
# # Could you do this in-place?
# class Solution:
#     def rotate(self, matrix):
#         n = len(matrix)
#         for i in xrange(n):
#             for j in xrange(n-1-i):
#                 matrix[i][j], matrix[n-1-j][n-1-i] = matrix[n-1-j][n-1-i], matrix[i][j]
#         for i in xrange(n/2):
#             for j in xrange(n):
#                 matrix[i][j], matrix[n-1-i][j] = matrix[n-1-i][j], matrix[i][j]
#         return matrix

# # Anagrams
# # Given an array of strings, return all groups of strings that are anagrams.
# # Note: All inputs will be in lower-case.
# class Solution:
#     def anagrams(self, strs):
#         dict = {}
#         for word in strs:
#             sortedWord = ''.join(sorted(word))
#             dict[sortedWord] = [word] if sortedWord not in dict else dict[sortedWord] + [word]
#         res = []
#         for item in dict:
#             if len(dict[item]) >= 2:
#                 res += dict[item]
#         return res

# # Pow(x, n)
# # Implement pow(x, n).
# class Solution:
#     def pow(self, x, n):
#         if n < 0:
#             return 1 / self.powRecu(x, -n)
#         return self.powRecu(x, n)
#     def powRecu(self, x, n):
#         if n == 0:
#             return 1.0
#         if n % 2 == 0:
#             return self.powRecu(x*x, n/2)
#         else:
#             return x * self.powRecu(x*x, n/2)

# # N-Queens
# # The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.
# # Given an integer n, return all distinct solutions to the n-queens puzzle.
# # Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space respectively.
# # For example,
# # There exist two distinct solutions to the 4-queens puzzle:
# class Solution:
#     def solveNQueens(self, n):
#         self.res = []
#         self.solve(n, 0, [-1 for i in xrange(n)])
#         return self.res
#     def solve(self, n, currQueenNum, board):
#         if currQueenNum == n:
#             oneAnswer = [['.' for j in xrange(n)] for i in xrange(n)]
#             for i in xrange(n): 
#                 oneAnswer[i][board[i]] = 'Q'
#                 oneAnswer[i] = ''.join(oneAnswer[i])
#             self.res.append(oneAnswer)
#             return
#         # try to put a Queen in (currQueenNum, 0), (currQueenNum, 1), ..., (currQueenNum, n-1)
#         for i in xrange(n):
#             valid = True  # test whether board[currQueenNum] can be i or not
#             for k in xrange(currQueenNum):
#                 # check column
#                 if board[k] == i: valid = False; break
#                 # check dianogal
#                 if abs(board[k] - i) == currQueenNum - k: valid = False; break
#             if valid:
#                 board[currQueenNum] = i
#                 self.solve(n, currQueenNum + 1, board)

# # N-Queens II
# # Follow up for N-Queens problem.
# # Now, instead outputting board configurations, return the total number of distinct solutions.
# class Solution:
#     # @return an integer
#     def totalNQueens(self, n):
#         self.res = 0
#         self.solve(n, 0, [-1 for i in xrange(n)])
#         return self.res
#     def solve(self, n, currQueenNum, board):
#         if currQueenNum == n: self.res += 1; return
#         # try to put a Queen in (currQueenNum, 0), (currQueenNum, 1), ..., (currQueenNum, n-1)
#         for i in xrange(n):
#             valid = True  # test whether board[currQueenNum] can be i or not
#             for k in xrange(currQueenNum):
#                 # check column
#                 if board[k] == i: valid = False; break
#                 # check diagonal
#                 if abs(board[k] - i) == currQueenNum - k: valid = False; break
#             if valid:
#                 board[currQueenNum] = i
#                 self.solve(n, currQueenNum + 1, board)

# # Maximum Subarray
# # Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
# # For example, given the array [−2,1,−3,4,−1,2,1,−5,4],
# # the contiguous subarray [4,−1,2,1] has the largest sum = 6.
# class Solution:
#     def maxSubArray(self, A):
#         length = len(A)
#         S = [0 for i in xrange(length)]
#         S[0] = A[0]
#         for i in xrange(1, length):
#             if S[i-1] > 0:
#                 S[i] = S[i-1] + A[i]
#             else:
#                 S[i] = A[i]
#         return max(S)

# # Spiral Matrix
# # Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
# # For example,
# # Given the following matrix:
# # [
# #  [ 1, 2, 3 ],
# #  [ 4, 5, 6 ],
# #  [ 7, 8, 9 ]
# # ]
# # You should return [1,2,3,6,9,8,7,4,5].
# class Solution:
#     def spiralOrder(self, matrix):
#         rowLen = len(matrix)
#         if rowLen == 0:
#         	return []       # Empty list
#         colLen = len(matrix[0])
#         result = []
#         bound = min(rowLen, colLen)
#         for layer in xrange(bound//2):
#             # Access the top line
#             result.extend(matrix[layer][layer:colLen-layer-1])
#             # Access the right line
#             result.extend([matrix[i][colLen-layer-1] for i in xrange(layer,rowLen-layer-1)])
#             # Access the bottom line
#             result.extend(matrix[rowLen-layer-1][colLen-layer-1:layer:-1])
#             # Access the left line
#             result.extend([matrix[i][layer] for i in xrange(rowLen-layer-1,layer,-1)])
#         # Maybe one line is remaining for access
#         if bound % 2 == 1:
#             if bound == rowLen:
#                 # The last horizontal line
#                 result.extend(matrix[bound//2][bound//2:colLen-bound//2])
#             else:
#                 # The last vertical line
#                 result.extend([matrix[i][bound//2] for i in xrange(bound//2,rowLen-bound//2)])
#         return result

# # Jump Game
# # Given an array of non-negative integers, you are initially positioned at the first index of the array.
# # Each element in the array represents your maximum jump length at that position.
# # Determine if you are able to reach the last index.
# # For example:
# # A = [2,3,1,1,4], return true.
# # A = [3,2,1,0,4], return false.
# class Solution:
#     def canJump(self, A):
#         lenA = len(A)
#         canReach = 0
#         for i in xrange(lenA):
#             if i <= canReach:
#                 canReach = max(canReach, i + A[i])
#                 if canReach >= lenA - 1: return True
#         return False

# # Merge Intervals
# # Given a collection of intervals, merge all overlapping intervals.
# # For example,
# # Given [1,3],[2,6],[8,10],[15,18],
# # return [1,6],[8,10],[15,18].
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e
# class Solution:
#     def merge(self, intervals):
#         if intervals == []: return []
#         intervals.sort(key = lambda x:x.start)
#         res = []; prev = Start = End = None;
#         for curr in intervals:
#             if prev:
#                 if curr.start <= End:
#                     End = max(End, curr.end)
#                 else:
#                     res.append([Start, End])
#                     Start = curr.start; End = curr.end
#             else:
#                 Start = curr.start; End = curr.end
#             prev = curr
#         res.append([Start, End])
#         return res

# # Insert Interval
# # Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
# # You may assume that the intervals were initially sorted according to their start times.
# # Example 1:
# # Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].
# # Example 2:
# # Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].
# # This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e
# class Solution:
#     def insert(self, intervals, newInterval):
#     	intervals.append(newInterval)
#         intervals.sort(key = lambda x:x.start)
#         res = []; prev = Start = End = None;
#         for curr in intervals:
#             if prev:
#                 if curr.start <= End:
#                     End = max(End, curr.end)
#                 else:
#                     res.append([Start, End])
#                     Start = curr.start; End = curr.end
#             else:
#                 Start = curr.start; End = curr.end
#             prev = curr
#         res.append([Start, End])
#         return res

# # Length of Last Word
# # Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.
# # If the last word does not exist, return 0.
# # Note: A word is defined as a character sequence consists of non-space characters only.
# # For example, 
# # Given s = "Hello World",
# # return 5.
# class Solution:
#     def lengthOfLastWord(self, s):
#         s_split = s.split()
#         return 0 if len(s_split) == 0 else len(s_split[-1])


# # Spiral Matrix II
# # Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.
# # For example,
# # Given n = 3,
# # You should return the following matrix:
# # [
# #  [ 1, 2, 3 ],
# #  [ 8, 9, 4 ],
# #  [ 7, 6, 5 ]
# # ]
# class Solution:
#     def generateMatrix(self, n):
#         if n == 0:
#         	return []
#         result = [[0]*n for _ in xrange(n)]
#         current = 1
#         for layer in xrange(n//2):
#             # Fill the top line
#             for i in xrange(layer, n-layer-1):
#                 result[layer][i] = current
#                 current += 1
#             # Fill the right line
#             for i in xrange(layer,n-layer-1):
#                 result[i][n-layer-1] = current
#                 current += 1
#             # Fill the bottom line
#             for i in xrange(n-layer-1,layer,-1):
#                 result[n-layer-1][i] = current
#                 current += 1
#             # Fill the left line
#             for i in xrange(n-layer-1,layer,-1):
#                 result[i][layer] = current
#                 current += 1
#         # Fill the center
#         if n % 2 == 1:
#         	result[n//2][n//2] = current
#         return result

# # Permutation Sequence
# # The set [1,2,3,…,n] contains a total of n! unique permutations.
# # By listing and labeling all of the permutations in order,
# # We get the following sequence (ie, for n = 3):
# # "123"
# # "132"
# # "213"
# # "231"
# # "312"
# # "321"
# # Given n and k, return the kth permutation sequence.
# # Note: Given n will be between 1 and 9 inclusive.
# class Solution:
#     def getPermutation(self, n, k):
#         k -= 1 # count from 0
#         res = ''
#         factorial = 1
#         for i in xrange(1, n):
#             factorial *= i
#         num = [i for i in xrange(1, n + 1)]
#         for i in reversed(xrange(n)):
#             curr = num[k / factorial]
#             res += str(curr)
#             num.remove(curr)
#             if i != 0:
#                 k %= factorial
#                 factorial /= i
#         return res

# # Rotate List
# # Given a list, rotate the list to the right by k places, where k is non-negative.
# # For example:
# # Given 1->2->3->4->5->NULL and k = 2,
# # return 4->5->1->2->3->NULL.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class Solution:
#     def rotateRight(self, head, k):
#         assert k >= 0
#         if head == None:
#         	return head
#         listLen = 0
#         temp = head
#         while temp != None:
#             temp = temp.next
#             listLen += 1
#         k = k % listLen
#         if k == 0:
#         	return head
#         former = latter = head
#         for _ in xrange(k):
#             latter = latter.next            
#         while latter.next != None:
#             latter = latter.next
#             former = former.next
#         newHead = former.next
#         latter.next = head
#         former.next = None
#         return newHead

# # Unique Paths 
# # A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
# # The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
# # How many possible unique paths are there?
# class Solution:
#     def uniquePaths(self, m, n):
#         N = m - 1 + n - 1
#         K = min(m, n) - 1
#         # calculate C(N, K)
#         res = 1
#         for i in xrange(K):
#             res = res * (N - i) / (i + 1)
#         return res

# # Unique Paths II
# # Follow up for "Unique Paths":
# # Now consider if some obstacles are added to the grids. How many unique paths would there be?
# # An obstacle and empty space is marked as 1 and 0 respectively in the grid.
# # For example,
# # There is one obstacle in the middle of a 3x3 grid as illustrated below.
# # [
# #   [0,0,0],
# #   [0,1,0],
# #   [0,0,0]
# # ]
# # The total number of unique paths is 2.
# class Solution:
#     def uniquePathsWithObstacles(self, grid):
#         rows = len(grid)
#         cols = len(grid[0])         
#         dp = [[0 for j in xrange(cols)] for i in xrange(rows)]
#         for i in xrange(cols):
#             if grid[0][i] == 0: dp[0][i] = 1
#             else: break
#         for i in xrange(rows):
#             if grid[i][0] == 0: dp[i][0] = 1
#             else: break         
#         # dynamic programming
#         for i in xrange(1, rows):
#             for j in xrange(1, cols):
#                 dp[i][j] = 0 if grid[i][j] == 1 else dp[i - 1][j] + dp[i][j - 1]
#         return dp[rows - 1][cols - 1]

# # Minimum Path Sum
# # Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.
# # Note: You can only move either down or right at any point in time.
# class Solution:
#     def minPathSum(self, grid):
#         rows = len(grid)
#         cols = len(grid[0])
#         dp = [[0 for j in xrange(cols)] for i in xrange(rows)]
#         dp[0][0] = grid[0][0]
#         for i in xrange(1, cols):
#             dp[0][i] = dp[0][i - 1] + grid[0][i]
#         for i in xrange(1, rows):
#             dp[i][0] = dp[i - 1][0] + grid[i][0]
#         # dynamic programming
#         for i in xrange(1, rows):
#             for j in xrange(1, cols):
#                 dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
#         return dp[rows - 1][cols - 1]

# # Valid Number
# # Validate if a given string is numeric.
# # Some examples:
# # "0" => true
# # " 0.1 " => true
# # "abc" => false
# # "1 a" => false
# # "2e10" => true
# # Note: It is intended for the problem statement to be ambiguous. You should gather all requirements up front before implementing one.
# # regular expression: "^\s*[\+\-]?((\d+(\.\d*)?)|\.\d+)([eE][+-]?\d+)?\s*$"
# class Solution:
#     def isNumber(self, s):
#         INVALID=0; SPACE=1; SIGN=2; DIGIT=3; DOT=4; EXPONENT=5;
#         #0invalid,1space,2sign,3digit,4dot,5exponent,6num_inputs
#         transitionTable=[[-1,  0,  3,  1,  2, -1],    #0 no input or just spaces 
#                          [-1,  8, -1,  1,  4,  5],    #1 input is digits 
#                          [-1, -1, -1,  4, -1, -1],    #2 no digits in front just Dot 
#                          [-1, -1, -1,  1,  2, -1],    #3 sign 
#                          [-1,  8, -1,  4, -1,  5],    #4 digits and dot in front 
#                          [-1, -1,  6,  7, -1, -1],    #5 input 'e' or 'E' 
#                          [-1, -1, -1,  7, -1, -1],    #6 after 'e' input sign 
#                          [-1,  8, -1,  7, -1, -1],    #7 after 'e' input digits 
#                          [-1,  8, -1, -1, -1, -1]]    #8 after valid input input space
#         state=0; i=0
#         while i<len(s):
#             inputtype = INVALID
#             if s[i]==' ': inputtype=SPACE
#             elif s[i]=='-' or s[i]=='+': inputtype=SIGN
#             elif s[i] in '0123456789': inputtype=DIGIT
#             elif s[i]=='.': inputtype=DOT
#             elif s[i]=='e' or s[i]=='E': inputtype=EXPONENT
#             state=transitionTable[state][inputtype]
#             if state==-1: return False
#             else: i+=1
#         return state == 1 or state == 4 or state == 7 or state == 8

# # Plus One
# # Given a non-negative number represented as an array of digits, plus one to the number.
# # The digits are stored such that the most significant digit is at the head of the list.
# class Solution:
#     def plusOne(self, digits):
#         carry = 0; length = len(digits)
#         for i in reversed(xrange(length)):
#             digits[i] = digits[i] + 1 + carry if i == length - 1 else digits[i] + carry
#             carry = 1 if digits[i] == 10 else 0
#             if digits[i] == 10: digits[i] = 0
#         return [1] + digits if carry == 1 else digits

# # Add Binary
# # Given two binary strings, return their sum (also a binary string).
# # For example,
# # a = "11"
# # b = "1"
# # Return "100".
# class Solution:
#     def addBinary(self, a, b):
#         length_a = len(a)
#         length_b = len(b)
#         if length_a > length_b:
#             b = '0' * (length_a - length_b) + b
#             length = length_a
#         else:
#             a = '0' * (length_b - length_a) + a
#             length = length_b
#         a = a[::-1]
#         b = b[::-1]
#         Sum = ''
#         carry = 0
#         for i in xrange(length):
#             tmp = ord(a[i]) - 48 + ord(b[i]) - 48 + carry
#             Sum += str(tmp % 2)
#             carry = tmp / 2
#         if carry == 1:
#             Sum += '1'
#         return Sum[::-1]

# # Text Justification
# # Given an array of words and a length L, format the text such that each line has exactly L characters and is fully (left and right) justified.
# # You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly L characters.
# # Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
# # For the last line of text, it should be left justified and no extra space is inserted between words.
# # For example,
# # words: ["This", "is", "an", "example", "of", "text", "justification."]
# # L: 16.
# # Return the formatted lines as:
# # [
# #    "This    is    an",
# #    "example  of text",
# #    "justification.  "
# # ]
# class Solution:
#     def fullJustify(self, words, L):
#         res=[]
#         i=0
#         while i<len(words):
#             size=0; begin=i
#             while i<len(words):
#                 newsize=len(words[i]) if size==0 else size+len(words[i])+1
#                 if newsize<=L: size=newsize
#                 else: break
#                 i+=1
#             spaceCount=L-size
#             if i-begin-1>0 and i<len(words):
#                 everyCount=spaceCount/(i-begin-1)
#                 spaceCount%=i-begin-1
#             else:
#                 everyCount=0
#             j=begin
#             while j<i:
#                 if j==begin: s=words[j]
#                 else:
#                     s+=' '*(everyCount+1)
#                     if spaceCount>0 and i<len(words):
#                         s+=' '
#                         spaceCount-=1
#                     s+=words[j]
#                 j+=1
#             s+=' '*spaceCount
#             res.append(s)
#         return res

# # Sqrt(x)
# # Implement int sqrt(int x).
# # Compute and return the square root of x.
# class Solution:
#     def sqrt(self, x):
#         left = 0
#         right = 46340 # sqrt(C MAX_INT 2147483647)=46340.950001
#         while left <= right:
#             mid = (left + right) / 2
#             if mid ** 2 <= x < (mid + 1) ** 2:
#                 return mid
#             elif x < mid ** 2:
#                 right = mid - 1
#             else:
#                 left = mid + 1

# # Climbing Stairs
# # You are climbing a stair case. It takes n steps to reach to the top.
# # Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# class Solution:
#     def climbStairs(self, n):
#         dp = [0, 1, 2]
#         i = 3
#         while i <= n:
#             dp.append(dp[i-1] + dp[i-2])
#             i += 1
#         return dp[n]

# # Simplify Path
# # Given an absolute path for a file (Unix-style), simplify it.
# # For example,
# # path = "/home/", => "/home"
# # path = "/a/./b/../../c/", => "/c"
# class Solution:
#     def simplifyPath(self, path):
#         stack, tokens = [], path.split("/")
#         for token in tokens:
#             if token == ".." and stack:
#                 stack.pop()
#             elif token != ".." and token != "." and token:
#                 stack.append(token)
#         return "/" + "/".join(stack)

# # Edit Distance
# # Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)
# # You have the following 3 operations permitted on a word:
# # a) Insert a character
# # b) Delete a character
# # c) Replace a character
# class Solution2:
#     def minDistance(self, word1, word2):        
#         distance = [[i] for i in xrange(len(word1) + 1)]
#         distance[0] = [j for j in xrange(len(word2) + 1)]
#         for i in xrange(1, len(word1) + 1):
#             for j in xrange(1, len(word2) + 1):
#                 insert = distance[i][j - 1] + 1
#                 delete = distance[i - 1][j] + 1
#                 replace = distance[i - 1][j - 1]
#                 if word1[i - 1] != word2[j - 1]:
#                     replace += 1
#                 distance[i].append(min(insert, delete, replace))
#         return distance[-1][-1]

# # Set Matrix Zeroes
# # Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.
# class Solution:
#     def setZeroes(self, matrix):
#         first_col = reduce(lambda acc, i: acc or matrix[i][0] == 0, xrange(len(matrix)), False)
#         first_row = reduce(lambda acc, j: acc or matrix[0][j] == 0, xrange(len(matrix[0])), False)
#         for i in xrange(1, len(matrix)):
#             for j in xrange(1, len(matrix[0])):
#                 if matrix[i][j] == 0:
#                     matrix[i][0], matrix[0][j] = 0, 0
#         for i in xrange(1, len(matrix)):
#             for j in xrange(1, len(matrix[0])):
#                 if matrix[i][0] == 0 or matrix[0][j] == 0:
#                     matrix[i][j] = 0
#         if first_col:
#             for i in xrange(len(matrix)):
#                 matrix[i][0] = 0
#         if first_row:
#             for j in xrange(len(matrix[0])):
#                 matrix[0][j] = 0

# # Search a 2D Matrix
# # Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
# # Integers in each row are sorted from left to right.
# # The first integer of each row is greater than the last integer of the previous row.
# # For example,
# # Consider the following matrix:
# # [
# #   [1,   3,  5,  7],
# #   [10, 11, 16, 20],
# #   [23, 30, 34, 50]
# # ]
# # Given target = 3, return true.
# class Solution:
#     def searchMatrix(self, matrix, target):
#         m = len(matrix)
#         n = len(matrix[0])
#         i, j = 0, m * n
#         while i < j:
#             mid = i + (j - i) / 2
#             val = matrix[mid / n][mid % n]
#             if val == target:
#                 return True
#             elif val < target:
#                 i = mid + 1
#             else:
#                 j = mid
#         return False

# # Sort Colors
# # Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.
# # Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
# class Solution:
#     def sortColors(self, A):
#         i, last_zero, first_two = 0, -1, len(A)
#         while i < first_two:
#             if A[i] == 0:
#                 last_zero += 1
#                 A[last_zero], A[i] = A[i], A[last_zero]
#             elif A[i] == 2:
#                 first_two -= 1
#                 A[first_two], A[i] = A[i], A[first_two]
#                 i -= 1
#             i += 1

# # Minimum Window Substring
# # Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).
# # For example,
# # S = "ADOBECODEBANC"
# # T = "ABC"
# # Minimum window is "BANC".
# class Solution:
#     # @return a string
#     def minWindow(self, S, T):
#         current_count = [0 for i in xrange(52)]
#         expected_count = [0 for i in xrange(52)]
#         for char in T:
#             expected_count[ord(char) - ord('a')] += 1
#         i, count, start, min_width, min_start = 0, 0, 0, float("inf"), 0
#         while i < len(S):
#             current_count[ord(S[i]) - ord('a')] += 1
#             if current_count[ord(S[i]) - ord('a')] <= expected_count[ord(S[i]) - ord('a')]:
#                 count += 1
#             if count == len(T):
#                 while expected_count[ord(S[start]) - ord('a')] == 0 or\
#                       current_count[ord(S[start]) - ord('a')] > expected_count[ord(S[start]) - ord('a')]:
#                     current_count[ord(S[start]) - ord('a')] -= 1
#                     start += 1            
#                 if min_width > i - start + 1:
#                     min_width = i - start + 1
#                     min_start = start
#             i += 1        
#         if min_width == float("inf"):
#             return ""
#         return S[min_start:min_start + min_width]

# # Combinations
# # Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
# # For example,
# # If n = 4 and k = 2, a solution is:
# # [
# #   [2,4],
# #   [3,4],
# #   [2,3],
# #   [1,2],
# #   [1,3],
# #   [1,4],
# # ]
# class Solution:
#     def combine(self, n, k):
#         result = []
#         self.combineRecu(n, result, 0, [], k)
#         return result
#     def combineRecu(self, n, result, start, intermediate, k):
#         if k == 0:
#             result.append(intermediate[:])
#         for i in xrange(start, n):
#             intermediate.append(i + 1)
#             self.combineRecu(n, result, i + 1, intermediate, k - 1)
#             intermediate.pop()

# # Subsets
# # Given a set of distinct integers, S, return all possible subsets.
# # Note:
# # Elements in a subset must be in non-descending order.
# # The solution set must not contain duplicate subsets.
# # For example,
# # If S = [1,2,3], a solution is:
# # [
# #   [3],
# #   [1],
# #   [2],
# #   [1,2,3],
# #   [1,3],
# #   [2,3],
# #   [1,2],
# #   []
# # ]
# class Solution:
#     def subsets(self, S):
#         return self.subsetsRecu([], sorted(S))
#     def subsetsRecu(self, cur, S):
#         if not S:
#             return [cur]
#         return self.subsetsRecu(cur, S[1:]) + self.subsetsRecu(cur + [S[0]], S[1:])

# # Word Search
# # Given a 2D board and a word, find if the word exists in the grid.
# # The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.
# # For example,
# # Given board =
# # [
# #   ["ABCE"],
# #   ["SFCS"],
# #   ["ADEE"]
# # ]
# # word = "ABCCED", -> returns true,
# # word = "SEE", -> returns true,
# # word = "ABCB", -> returns false.
# class Solution:
#     def exist(self, board, word):
#         visited = [[False for j in xrange(len(board[0]))] for i in xrange(len(board))]
#         for i in xrange(len(board)):
#             for j in xrange(len(board[0])):
#                 if self.existRecu(board, word, 0, i, j, visited):
#                     return True
#         return False
#     def existRecu(self, board, word, cur, i, j, visited):
#         if cur == len(word):
#             return True
#         if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or visited[i][j] or board[i][j] != word[cur]:
#             return False
#         visited[i][j] = True
#         result = self.existRecu(board, word, cur + 1, i + 1, j, visited) or\
#                  self.existRecu(board, word, cur + 1, i - 1, j, visited) or\
#                  self.existRecu(board, word, cur + 1, i, j + 1, visited) or\
#                  self.existRecu(board, word, cur + 1, i, j - 1, visited)         
#         visited[i][j] = False 
#         return result

# # Remove Duplicates from Sorted Array II
# # Follow up for "Remove Duplicates":
# # What if duplicates are allowed at most twice?
# # For example,
# # Given sorted array A = [1,1,1,2,2,3],
# # Your function should return length = 5, and A is now [1,1,2,2,3].
# class Solution:
#     def removeDuplicates(self, A):
#         if not A:
#             return 0
#         last, i, same = 0, 1, False
#         while i < len(A):
#             if A[last] != A[i] or not same:
#                 same = A[last] == A[i]
#                 last += 1
#                 A[last] = A[i]
#             i += 1
#         return last + 1

# # Search in Rotated Sorted Array II
# # Follow up for "Search in Rotated Sorted Array":
# # What if duplicates are allowed?
# # Would this affect the run-time complexity? How and why?
# # Write a function to determine if a given target is in the array.
# class Solution:
#     def search(self, A, target):
#         low, high = 0, len(A)
#         while low < high:
#             mid = low + (high - low) / 2
#             if A[mid] == target:
#                 return True
#             if A[low] < A[mid]:
#                 if A[low] <= target and target < A[mid]:
#                     high = mid
#                 else:
#                     low = mid + 1
#             elif A[low] > A[mid]:
#                 if A[mid] < target and target <= A[high - 1]:
#                     low = mid + 1
#                 else:
#                     high = mid
#             else:
#                 low += 1
#         return False

# # Remove Duplicates from Sorted List II
# # Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.
# # For example,
# # Given 1->2->3->3->4->4->5, return 1->2->5.
# # Given 1->1->1->2->3, return 2->3.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class Solution:
#     def deleteDuplicates(self, head):
#         dummy = ListNode(0)
#         dummy.next = head
#         current = dummy
#         while current.next:
#             next = current.next
#             while next.next and next.next.val == next.val:
#                 next = next.next
#             if current.next is not next:
#                 current.next = next.next
#             else:
#                 current = current.next
#         return dummy.next

# # Remove Duplicates from Sorted List
# # Given a sorted linked list, delete all duplicates such that each element appear only once.
# # For example,
# # Given 1->1->2, return 1->2.
# # Given 1->1->2->3->3, return 1->2->3.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class Solution:
#     def deleteDuplicates(self, head):
#         current = head
#         while current and current.next:
#             next = current.next
#             if current.val == next.val:
#                 current.next = current.next.next
#             else:
#                 current = next
#         return head

# # Largest Rectangle in Histogram
# # Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.
# class Solution:
#     def largestRectangleArea(self, height):
#         increasing, area, i = [], 0, 0
#         while i <= len(height):
#             if not increasing or (i < len(height) and height[i] > height[increasing[-1]]):
#                 increasing.append(i)
#                 i += 1
#             else:
#                 last = increasing.pop()
#                 if not increasing:
#                     area = max(area, height[last] * i)
#                 else:
#                     area = max(area, height[last] * (i - increasing[-1] - 1 ))
#         return area

# # Maximal Rectangle
# # Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing all ones and return its area.
# class Solution:
#     def maximalRectangle(self, matrix):
#         if not matrix:
#             return 0
#         result = 0
#         m = len(matrix)
#         n = len(matrix[0])
#         L = [0 for _ in xrange(n)]
#         H = [0 for _ in xrange(n)]
#         R = [n for _ in xrange(n)]
#         for i in xrange(m):
#             left = 0
#             for j in xrange(n):
#                 if matrix[i][j] == '1':
#                     L[j] = max(L[j], left)
#                     H[j] += 1
#                 else:
#                     L[j] = 0
#                     H[j] = 0
#                     R[j] = n
#                     left = j + 1
#             right = n
#             for j in reversed(xrange(n)):
#                 if matrix[i][j] == '1':
#                     R[j] = min(R[j], right)
#                     result = max(result, H[j] * (R[j] - L[j]))
#                 else:
#                     right = j
#         return result

# # Partition List
# # Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
# # You should preserve the original relative order of the nodes in each of the two partitions.
# # For example,
# # Given 1->4->3->2->5->2 and x = 3,
# # return 1->2->2->4->3->5.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class Solution:
#     def partition(self, head, x):
#         dummySmaller, dummyGreater = ListNode(-1), ListNode(-1)
#         smaller, greater = dummySmaller, dummyGreater
#         while head:
#             if head.val < x:
#                 smaller.next = head
#                 smaller = smaller.next
#             else:
#                 greater.next = head
#                 greater = greater.next
#             head = head.next
#         smaller.next = dummyGreater.next
#         greater.next = None
        
# # Scramble String
# # Given a string s1, we may represent it as a binary tree by partitioning it to two non-empty substrings recursively.
# # Below is one possible representation of s1 = "great":
# #     great
# #    /    \
# #   gr    eat
# #  / \    /  \
# # g   r  e   at
# #            / \
# #           a   t
# # To scramble the string, we may choose any non-leaf node and swap its two children.
# # For example, if we choose the node "gr" and swap its two children, it produces a scrambled string "rgeat".
# #     rgeat
# #    /    \
# #   rg    eat
# #  / \    /  \
# # r   g  e   at
# #            / \
# #           a   t
# # We say that "rgeat" is a scrambled string of "great".
# # Similarly, if we continue to swap the children of nodes "eat" and "at", it produces a scrambled string "rgtae".
# #     rgtae
# #    /    \
# #   rg    tae
# #  / \    /  \
# # r   g  ta  e
# #        / \
# #       t   a
# # We say that "rgtae" is a scrambled string of "great".
# # Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1.
# class Solution:
#     def isScramble(self, s1, s2):
#         if not s1 or not s2 or len(s1) != len(s2):
#             return False
#         if not s1:
#             return True
#         result = [[[False for j in xrange(len(s2))] for i in xrange(len(s1))] for n in xrange(len(s1) + 1)]
#         for i in xrange(len(s1)):
#             for j in xrange(len(s2)):
#                 if s1[i] == s2[j]:
#                     result[1][i][j] = True
#         for n in xrange(2, len(s1) + 1):
#             for i in xrange(len(s1) - n + 1):
#                 for j in xrange(len(s2) - n + 1):
#                     for k in xrange(1, n):
#                         if result[k][i][j] and result[n - k][i + k][j + k] or\
#                            result[k][i][j + n - k] and result[n - k][i + k][j]:
#                             result[n][i][j] = True
#                             break
#         return result[n][0][0]

# # Merge Sorted Array
# # Given two sorted integer arrays A and B, merge B into A as one sorted array.
# class Solution:
#     def merge(self, A, m, B, n):
#         last, i, j = m + n - 1, m - 1, n - 1
#         while i >= 0 and j >= 0:
#             if A[i] > B[j]:
#                 A[last] = A[i]
#                 last, i = last - 1, i - 1
#             else:
#                 A[last] = B[j]
#                 last, j = last - 1, j - 1
#         while j >= 0:
#                 A[last] = B[j]
#                 last, j = last - 1, j - 1

# # Gray Code
# # The gray code is a binary numeral system where two successive values differ in only one bit.
# # Given a non-negative integer n representing the total number of bits in the code, print the sequence of gray code. A gray code sequence must begin with 0.
# # For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:
# # 00 - 0
# # 01 - 1
# # 11 - 3
# # 10 - 2
# class Solution:
#     def grayCode(self, n):
#         result = [0]
#         for i in xrange(0, n):
#             for n in reversed(result):
#                 result.append(1 << i | n)
#         return result

# # Subsets II
# # Given a collection of integers that might contain duplicates, S, return all possible subsets.
# class Solution:
#     def subsetsWithDup(self, S):
#         result = []
#         self.subsetsWithDupRecu(result, [], sorted(S))
#         return result
#     def subsetsWithDupRecu(self, result, cur, S):
#         if len(S) == 0 and cur not in result:
#             result.append(cur)
#         elif S:
#             self.subsetsWithDupRecu(result, cur, S[1:])
#             self.subsetsWithDupRecu(result, cur + [S[0]], S[1:])

# # Decode Ways
# # A message containing letters from A-Z is being encoded to numbers using the following mapping:
# # 'A' -> 1
# # 'B' -> 2
# # ...
# # 'Z' -> 26
# # Given an encoded message containing digits, determine the total number of ways to decode it.
# # For example,
# # Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).
# # The number of ways decoding "12" is 2.
# class Solution:
#     def numDecodings(self, s):
#         if len(s) == 0 or s[0] == '0':
#             return 0
#         prev, prev_prev = 1, 0
#         for i in range(len(s)):
#             current = 0
#             if s[i] != '0':
#                 current = prev
#             if i > 0 and (s[i - 1] == '1' or (s[i - 1] == '2' and s[i] <= '6')):
#                 current += prev_prev
#             prev, prev_prev = current, prev
#         return prev

# # Reverse Linked List II
# # Reverse a linked list from position m to n. Do it in-place and in one-pass.
# # For example:
# # Given 1->2->3->4->5->NULL, m = 2 and n = 4,
# # return 1->4->3->2->5->NULL.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class Solution:
#     def reverseBetween(self, head, m, n):
#         diff, dummy, cur = n - m + 1, ListNode(-1), head
#         dummy.next = head
#         last_unswapped = dummy
#         while cur and m > 1:
#             cur, last_unswapped, m = cur.next, cur, m - 1
#         prev, first_swapped = last_unswapped,  cur
#         while cur and diff > 0:
#             cur.next, prev, cur, diff = prev, cur, cur.next, diff - 1
#         last_unswapped.next, first_swapped.next = prev, cur
#         return dummy.next

# # Restore IP Addresses
# # Given a string containing only digits, restore it by returning all possible valid IP address combinations.
# # For example:
# # Given "25525511135",
# # return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
# class Solution:
#     def restoreIpAddresses(self, s):
#         result = []
#         self.restoreIpAddressesRecur(result, s, 0, "", 0)
#         return result        
#     def restoreIpAddressesRecur(self, result, s, start, current, dots):
#         if (4-dots)*3 < len(s)-start or (4-dots) > len(s)-start:
#             return
#         if start == len(s) and dots == 4:
#             result.append(current[:-1])
#         else:
#             for i in xrange(start, start + 3):
#                 if len(s) > i and self.isValid(s[start:i + 1]):
#                     current += s[start:i + 1] + '.'
#                     self.restoreIpAddressesRecur(result, s, i + 1, current, dots + 1)
#                     current = current[:-(i - start + 2)]
#     def isValid(self, s):
#         if len(s) == 0 or (s[0] == "0" and s != "0"):
#             return False
#         return int(s) < 256

# # Binary Tree Inorder Traversal
# # Given a binary tree, return the inorder traversal of its nodes' values.
# # For example:
# # Given binary tree {1,#,2,3},
# #    1
# #     \
# #      2
# #     /
# #    3
# # return [1,3,2].
# # Note: Recursive solution is trivial, could you do it iteratively?
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# class Solution:
#     def inorderTraversal(self, root):
#         result, prev, cur = [], None, root
#         while cur:
#             if cur.left is None:
#                 result.append(cur.val)
#                 prev = cur
#                 cur = cur.right
#             else:
#                 node = cur.left
#                 while node.right and node.right != cur:
#                     node = node.right
#                 if node.right is None:
#                     node.right = cur
#                     cur = cur.left
#                 else:
#                     result.append(cur.val)
#                     node.right = None
#                     prev = cur
#                     cur = cur.right
#         return result
# class Solution2:
#     def inorderTraversal(self, root):
#         result, stack, current, last_traversed = [], [], root, None
#         while stack or current:
#             if current:
#                 stack.append(current)
#                 current = current.left
#             else:
#                 parent = stack[-1]
#                 if parent.right in (None, last_traversed):
#                     if parent.right is None:
#                         result.append(parent.val)
#                     last_traversed= stack.pop()
#                 else:
#                     result.append(parent.val)
#                     current = parent.right
#         return result
# class Solution3:
#     def inorderTraversal(self, root):
#         result, stack, current, last_traversed = [], [], root, None
#         while stack or current:
#             if current:
#                 stack.append(current)
#                 current = current.left
#             else:
#                 current = stack[-1]
#                 stack.pop()
#                 result.append(current.val)
#                 current = current.right
#         return result

# # Unique Binary Search Trees II
# # Given n, generate all structurally unique BST's (binary search trees) that store values 1...n.
# # For example,
# # Given n = 3, your program should return all 5 unique BST's shown below.
# #    1         3     3      2      1
# #     \       /     /      / \      \
# #      3     2     1      1   3      2
# #     /     /       \                 \
# #    2     1         2                 3
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# class Solution:
#     def generateTrees(self, n):
#         return self.generateTreesRecu(1, n)
#     def generateTreesRecu(self, low, high):
#         result = []
#         if low > high:
#             result.append(None)
#         for i in xrange(low, high+1):
#             left = self.generateTreesRecu(low, i-1)
#             right = self.generateTreesRecu(i+1, high)
#             for j in left:
#                 for k in right:
#                     cur = TreeNode(i)
#                     cur.left = j
#                     cur.right = k
#                     result.append(cur)
#         return result

# # Unique Binary Search Trees
# # Given n, how many structurally unique BST's (binary search trees) that store values 1...n?
# # For example,
# # Given n = 3, there are a total of 5 unique BST's.
# #    1         3     3      2      1
# #     \       /     /      / \      \
# #      3     2     1      1   3      2
# #     /     /       \                 \
# #    2     1         2                 3
# class Solution:
#     def numTrees(self, n):
#         counts = [1, 1]
#         for i in xrange(2, n + 1):
#             count = 0
#             for j in xrange(i):
#                 count += counts[j] * counts[i - j - 1]
#             counts.append(count)
#         return counts[-1]

# # Interleaving String 
# # Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.
# # For example,
# # Given:
# # s1 = "aabcc",
# # s2 = "dbbca",
# # When s3 = "aadbbcbcac", return true.
# # When s3 = "aadbbbaccc", return false.
# class Solution:
#     def isInterleave(self, s1, s2, s3):
#         if len(s1) + len(s2) != len(s3):
#             return False
#         match = [[False for i in xrange(len(s2) + 1)] for j in xrange(len(s1) + 1)]
#         match[0][0] = True
#         for i in xrange(1, len(s1) + 1):
#             match[i][0] = match[i - 1][0] and s1[i - 1] == s3[i - 1]
#         for j in xrange(1, len(s2) + 1):
#             match[0][j] = match[0][j - 1] and s2[j - 1] == s3[j - 1]
#         for i in xrange(1, len(s1) + 1):
#             for j in xrange(1, len(s2) + 1):
#                 match[i][j] = (match[i - 1][j] and s1[i - 1] == s3[i + j - 1]) or (match[i][j - 1] and s2[j - 1] == s3[i + j - 1])
#         return match[-1][-1]

# # Validate Binary Search Tree
# # Given a binary tree, determine if it is a valid binary search tree (BST).
# # Assume a BST is defined as follows:
# # The left subtree of a node contains only nodes with keys less than the node's key.
# # The right subtree of a node contains only nodes with keys greater than the node's key.
# # Both the left and right subtrees must also be binary search trees.
# class Solution:
#     def isValidBST(self, root):
#         return self.isValidBSTRecu(root, float("-inf"), float("inf"))
#     def isValidBSTRecu(self, root, low, high):
#         if root is None:
#             return True
#         return low < root.val and root.val < high \
#             and self.isValidBSTRecu(root.left, low, root.val) \
#             and self.isValidBSTRecu(root.right, root.val, high)

# # Recover Binary Search Tree
# # Two elements of a binary search tree (BST) are swapped by mistake.
# # Recover the tree without changing its structure.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# class Solution:
#     def recoverTree(self, root):
#         return self.MorrisTraversal(root)
#     def MorrisTraversal(self, root):
#         if root is None:
#             return
#         broken = [None, None]
#         pre, cur = None, root
#         while cur:
#             if cur.left is None:
#                 self.detectBroken(broken, pre, cur)
#                 pre = cur
#                 cur = cur.right
#             else:
#                 node = cur.left
#                 while node.right and node.right != cur:
#                     node = node.right
#                 if node.right is None:
#                     node.right =cur
#                     cur = cur.left
#                 else:
#                     self.detectBroken(broken, pre, cur)
#                     node.right = None
#                     pre = cur
#                     cur = cur.right
#         broken[0].val, broken[1].val = broken[1].val, broken[0].val
#         return root
#     def detectBroken(self, broken, pre, cur):
#         if pre and pre.val > cur.val:
#             if broken[0] is None:
#                 broken[0] = pre
#             broken[1] = cur

# # Same Tree
# # Given two binary trees, write a function to check if they are equal or not.
# # Two binary trees are considered equal if they are structurally identical and the nodes have the same value.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# class Solution:
#     def isSameTree(self, p, q):
#         if p is None and q is None:
#             return True
#         if p is not None and q is not None:
#             return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
#         return False

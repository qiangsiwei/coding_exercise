# -*- coding: utf-8 -*- 

# FrogJmp
# Count minimal number of jumps from position X to Y.
def solution(X, Y, D):
    distance=Y-X
    if distance % D == 0: 
        return distance/D
    else:
        return distance/D +1

# PermMissingElem
# Find the missing element in a given permutation.
def solution(A):
    xor_sum = 0
    for index in range(0, len(A)):
        xor_sum = xor_sum^A[index]^(index+1)
    return xor_sum^(length+1)

# TapeEquilibrium
# Minimize the value |(A[0] + ... + A[P-1]) - (A[P] + ... + A[N-1])|.
def solution(A):
    head, tail = A[0], sum(A[1:])
    min_dif = abs(head - tail)
    for index in range(1, len(A)-1):
        head += A[index]
        tail -= A[index]
        if abs(head-tail) < min_dif:
            min_dif = abs(head-tail)
    return min_dif

# PermCheck
# Check whether array A is a permutation.
def solution(A):
    counter = [0]*len(A)
    for element in A:
        if not 1 <= element <= len(A):
            return 0
        else:
            if counter[element-1] != 0:
                return 0
            else:
                counter[element-1] += 1
    return 1

# FrogRiverOne
# Find the earliest time when a frog can jump to the other side of a river.
def solution(X, A):
    covered_time, uncovered = [-1]*X, X
    for index in range(0,len(A)):
        if covered_time[A[index]-1] != -1:
            continue
        else:
            covered_time[A[index]-1] = index
            uncovered -= 1
            if uncovered == 0:
                return index
    return -1
    `
# MaxCounters
# Calculate the values of counters after applying all alternating operations: 
# increase counter by 1; set value of all counters to current maximum.
def solution(N, A):
    result, max_counter, current_max = [0]*N, 0, 0
    for command in A:
        if 1 <= command <= N:
            if max_counter > result[command-1]:
                result[command-1] = max_counter
            result[command-1] += 1
            if current_max < result[command-1]:
                current_max = result[command-1]
        else:
            max_counter = current_max
    for index in range(0,N):
        if result[index] < max_counter:
            result[index] = max_counter
    return result

# MissingInteger
# Find the minimal positive integer not occurring in a given sequence.
def solution(A):
    occurrence = [False]*(len(A)+1)
    for item in A:
        if 1 <= item <= len(A)+1:
            occurrence[item-1] = True
    for index in xrange(len(A)+1):
        if occurrence[index] == False:
            return index+1
    return -1

# PassingCars
# Count the number of passing cars on the road.
def solution(A):
    west = 0
    passing = 0
    for index in range(len(A)-1,-1,-1):
        if A[index] == 0:
            passing += west
            if passing > 1000000000:
                return -1
        else:
            west += 1
    return passing

# CountDiv
# Compute number of integers divisible by k in range [a..b].
def solution(A, B, K):
    if A%K == 0:
        return (B-A)/K+1
    else:
        return (B-(A-A%K))/K

# MinAvgTwoSlice
# Find the minimal average of any slice containing at least two elements.
def solution(A):
    min_avg_value = (A[0]+A[1])/2.0
    min_avg_pos = 0
    for index in xrange(0, len(A)-2):
        if (A[index]+A[index+1])/2.0 < min_avg_value:
            min_avg_value = (A[index]+A[index+1])/2.0
            min_avg_pos = index
        if (A[index]+A[index+1]+A[index+2])/3.0<min_avg_value:
            min_avg_value = (A[index]+A[index+1]+A[index+2])/3.0
            min_avg_pos = index
    if (A[-1]+A[-2])/2.0 < min_avg_value:
        min_avg_value = (A[-1]+A[-2])/2.0
        min_avg_pos = len(A)-2
    return min_avg_pos

# GenomicRangeQuery
# Find the minimal nucleotide from a range of sequence DNA.
def solution(S, P, Q):
    result = []
    DNA_len = len(S)
    mapping = {"A":1, "C":2, "G":3, "T":4}
    next_nucl = [[-1]*DNA_len, [-1]*DNA_len, [-1]*DNA_len, [-1]*DNA_len]
    next_nucl[mapping[S[-1]]-1][-1] = DNA_len-1
    for index in range(DNA_len-2,-1,-1):
        next_nucl[0][index] = next_nucl[0][index+1]
        next_nucl[1][index] = next_nucl[1][index+1]
        next_nucl[2][index] = next_nucl[2][index+1]
        next_nucl[3][index] = next_nucl[3][index+1]
        next_nucl[mapping[S[index]] - 1][index] = index
    for index in range(0,len(P)):
        if next_nucl[0][P[index]] != -1 and next_nucl[0][P[index]] <= Q[index]:
            result.append(1)
        elif next_nucl[1][P[index]] != -1 and next_nucl[1][P[index]] <= Q[index]:
            result.append(2)
        elif next_nucl[2][P[index]] != -1 and next_nucl[2][P[index]] <= Q[index]:
            result.append(3)
        else:
            result.append(4)
    return result

# MaxProductOfThree
# Maximize A[P] * A[Q] * A[R] for any triplet (P, Q, R).
def solution(A):
    A.sort()
    return max(A[0]*A[1]*A[-1], A[-1]*A[-2]*A[-3])

# Distinct
# Compute number of distinct values in an array.
def solution(A):
    if len(A) == 0: distinct =0
    else:
        distinct =1
        A.sort()
        for index in xrange(1, len(A)):
            if A[index] == A[index-1]:
                continue
            else:
                distinct += 1
    return distinct

# Triangle
# Determine whether a triangle can be built from a given set of edges.
def solution(A):
    if len(A) < 3:
        return 0
    A.sort()
    for index in xrange(0, len(A)-2):
        if A[index]+A[index+1] > A[index+2]:
            return 1
    return 0

# NumberOfDiscIntersections
# Compute intersections between sequence of discs.
def solution(A):
    discs_count = len(A)
    range_upper = [0]*discs_count
    range_lower = [0]*discs_count
    for index in xrange(0, discs_count):
        range_upper[index] = index + A[index]
        range_upper[index] = index - A[index]
    range_upper.sort()
    range_lower.sort()
    range_lower_index, intersect_count = 0, 0
    for range_upper_index in xrange(0, discs_count):
        while range_lower_index < discs_count and \
            range_upper[range_upper_index] >= range_lower[range_lower_index]:
            range_lower_index += 1
        intersect_count += range_lower_index-range_upper_index-1
        if intersect_count > 10000000:
            return -1
    return intersect_count

# Brackets
# Determine whether a given string of parentheses is properly nested.
def solution(S):
    matched, to_push = {"]":"[", "}":"{", ")": "("}, ["[", "{", "("]
    stack = []
    for element in S:
        if element in to_push:
            stack.append(element)
        else:
            if len(stack) == 0:
                return 0
            elif matched[element] != stack.pop():
                return 0
    if len(stack) == 0:
        return 1
    else:
        return 0

# Nesting
# Determine whether given string of parentheses is properly nested.
def solution(S):
    parentheses = 0
    for element in S:
        if element == "(":
            parentheses += 1
        else:
            parentheses -= 1
            if parentheses < 0:
                return 0
    if parentheses == 0:
        return 1
    else:
        return 0

# Fish
# N voracious fish are moving along a river. Calculate how many fish are alive.
def solution(A, B):
    alive_count = 0
    downstream = []
    downstream_count = 0
    for index in xrange(len(A)):
        if B[index] == 1:
            downstream.append(A[index])
            downstream_count += 1
        else:
            while downstream_count != 0:
                if downstream[-1] < A[index]:
                    downstream_count -= 1
                    downstream.pop()
                else: 
                    break
            else:
                alive_count += 1
    alive_count += len(downstream)
    return alive_count

# StoneWall
# Cover "Manhattan skyline" using the minimum number of rectangles.
def solution(H):
    stack = []
    block_count = 0
    for height in H:
        while len(stack) != 0 and height < stack[-1]:
            stack.pop()
            block_count += 1
        if len(stack) == 0 or height > stack[-1]:
            stack.append(height)
    block_count += len(stack)
    return block_count

# EquiLeader
# Find the index S such that the leaders of the sequences A[0],A[1], ..., A[S] and A[S + 1], A[S + 2], ..., A[N - 1] are the same.
def solution(A):
    candidate, candidate_count = -1, 0
    for index in xrange(len(A)):
        if candidate_count == 0:
            candidate = A[index]
            candidate_count += 1
    else:
        if A[index] == candidate:
            candidate_count += 1
        else:
            candidate_count -= 1
    leader_count = len([number for number in A if number == candidate])
    if leader_count <= len(A)/2:
        return 0
    else:
        leader = candidate
    equi_leaders, leader_count_now = 0, 0
    for index in xrange(len(A)):
        if A[index] == leader:
            leader_count_now += 1
        if leader_count_now > (index+1)/2 and leader_count-leader_count_now > (len(A)-index-1)/2:
            equi_leaders += 1
    return equi_leaders

# Dominator
# Find an index of an array such that its value occurs at more than half of indices in the array.
def solution(A):
    candidate, candidate_count, candidate_index = -1, 0, -1
    for index in xrange(len(A)):
        if candidate_count == 0:
            candidate = A[index]
        candidate_index = index
        candidate_count += 1
    else:
        if A[index] == candidate:
            candidate_count += 1
        else:
            candidate_count -= 1
    if len([number for number in A if number == candidate]) <= len(A)/2:
        return -1
    else:
        return candidate_index

# MaxProfit
# Given a log of stock prices compute the maximum possible earning.
def solution(A):
    max_ending = max_slice = 0
    for i in xrange(1, len(A)):
        max_ending = max(0, max_ending + A[i] - A[i-1])
        max_slice = max(max_slice, max_ending)
    return max_slice

# MaxSliceSum
# Find a maximum sum of a compact subsequence of array elements.
def solution(A):
    max_ending = max_slice = 0
    max_A = max(A)
    if max_A > 0:
        for i in xrange(len(A)):
            max_ending = max(0, max_ending + A[i])
            max_slice = max(max_slice, max_ending)
    else:
        max_slice = max_A
    return max_slice

# MaxDoubleSliceSum
# Find the maximal sum of any double slice.
def solution(A):
    max_ending_here = [0]*len(A)
    max_ending_here_temp = 0
    for index in xrange(1, len(A)-1):
        max_ending_here_temp = max(0, A[index]+max_ending_here_temp)
        max_ending_here[index] = max_ending_here_temp
    max_beginning_here = [0]*len(A)
    max_beginning_here_temp = 0
    for index in xrange(len(A)-2, 0, -1):
        max_beginning_here_temp = max(0, A[index]+max_beginning_here_temp)
        max_beginning_here[index] = max_beginning_here_temp
    max_double_slice = 0
    for index in xrange(0, len(A)-2):
        max_double_slice = max(max_double_slice, max_ending_here[index]+max_beginning_here[index+2])
    return max_double_slice

# MinPerimeterRectangle
# Find the minimal perimeter of any rectangle whose area equals N.
def solution(N):
    from math import sqrt
    for i in xrange(int(sqrt(N)), 0, -1):
        if N % i == 0:
            return 2*(i+N/i)

# CountFactors
# Count factors of given number n.
def solution(N):
    candidate, result = 1, 0
    while candidate * candidate < N:
        if N % candidate == 0:
            result += 2
        candidate += 1
    if candidate * candidate == N:
      result += 1
    return result

# Peaks
# Divide an array into the maximum number of same((-))sized blocks, each of which should contain an index P such that A[P - 1] < A[P] > A[P + 1].
def solution(A):
    peaks = []
    for idx in xrange(1, len(A)-1):
        if A[idx-1] < A[idx] > A[idx+1]:
            peaks.append(idx)
    if len(peaks) == 0:
        return 0
    for size in xrange(len(peaks), 0, -1):
        if len(A) % size == 0:
            block_size = len(A)/size
            found, found_cnt = [False]*size, 0
            for peak in peaks:
                block_nr = peak/block_size
                if found[block_nr] == False:
                    found[block_nr] = True
                    found_cnt += 1
            if found_cnt == size:
                return size
    return 0

# Flags
# Find the maximum number of flags that can be set on mountain peaks.
def solution(A):
    from math import sqrt
    next_peak, peaks_count, first_peak = [-1]*len(A), 0, -1
    # Generate the information, where the next peak is.
    for index in xrange(len(A)-2, 0, -1):
        if A[index] > A[index+1] and A[index] > A[index-1]:
            next_peak[index] = index
            peaks_count += 1
            first_peak = index
        else:
            next_peak[index] = next_peak[index+1]
    if peaks_count < 2:
        return peaks_count
    max_flags = 1
    max_min_distance = int(sqrt(len(A)))
    for min_distance in xrange(max_min_distance+1, 1, -1):
        flags_used, flags_have = 1, min_distance-1
        pos = first_peak
        while flags_have > 0:
            if pos + min_distance >= len(A)-1:
                break
            pos = next_peak[pos+min_distance]
            if pos == -1:
                break
            flags_used += 1
            flags_have -= 1
        max_flags = max(max_flags, flags_used)
    return max_flags

# CountSemiprimes
# Count the semiprime numbers in the given range [a..b]
def solution(N, P, Q):
    from math import sqrt
    prime_table, prime, prime_count = [False]*2+[True]*(N-1), [], 0
    for element in xrange(2, int(sqrt(N))+1):
        if prime_table[element] == True:
            prime.append(element)
            prime_count += 1
            multiple = element*element
            while multiple <= N:
                prime_table[multiple] = False
                multiple += element
    for element in xrange(int(sqrt(N))+1, N+1):
        if prime_table[element] == True:
            prime.append(element)
            prime_count += 1
    semiprime = [0] * (N+1)
    for index_former in xrange(prime_count-1):
        for index_latter in xrange(index_former, prime_count):
            if prime[index_former]*prime[index_latter] > N:
                break
            semiprime[prime[index_former]*prime[index_latter]] = 1
    for index in xrange(1, N+1):
        semiprime[index] += semiprime[index-1]
    question_len = len(P)
    result = [0]*question_len
    for index in xrange(question_len):
        result[index] = semiprime[Q[index]] - semiprime[P[index]-1]
    return result

# CountNonDivisible
# Calculate the number of elements of an array that are not divisors of each element.
def solution(A):
    from math import sqrt
    A_len, A_max = len(A), max(A)
    count = {}
    for element in A:
        count[element] = count.get(element,0)+1
    divisors = {}
    for element in A:
        divisors[element] = [1]
    for divisor in xrange(2, int(sqrt(A_max))+1):
        multiple = divisor
        while multiple <= A_max:
            if multiple in divisors and not divisor in divisors[multiple]: 
                divisors[multiple].append(divisor)
            multiple += divisor
    for element in divisors:
        temp = [element/div for div in divisors[element]]
        temp = [item for item in temp if item not in divisors[element]]
        divisors[element].extend(temp)
    result = []
    for element in A:
        result.append(A_len-sum([count.get(div,0) for div in divisors[element]]))
    return result

# ChocolatesByNumbers
# There are N chocolates in a circle. Count the number of chocolates you will eat.
def gcd(a, b):
    if (a % b == 0):
        return b
    else:
        return gcd(b, a % b)
def solution(N, M):
    lcm = N * M / gcd(N, M)
    return lcm / M

# CommonPrimeDivisors
# Check whether two numbers have the same prime divisors.
def gcd(x, y):
    if x%y == 0:
        return y;
    else:
        return gcd(y, x%y)
def hasSamePrimeDivisors(x, y):
    gcd_value = gcd(x, y)
    while x != 1:
        x_gcd = gcd(x, gcd_value)
        if x_gcd == 1:
            break
        x /= x_gcd
    if x!=1: 
        return False
    while y != 1:
        y_gcd = gcd(y, gcd_value)
        if y_gcd == 1:
            break
        y /= y_gcd
    return y == 1
def solution(A, B):
    count = 0
    for x,y in zip(A,B):
        if hasSamePrimeDivisors(x,y):
            count += 1
    return count

# Ladder
# Count the number of different ways of climbing to the top of a ladder.
def solution(A, B):
    limit = len(A)                  # The possible largest N rungs
    result = [0] * len(A)           # The result for each query
    B = [(1<<item)-1 for item in B] # Pre-compute B for optimization
    # Compute the Fibonacci numbers for later use
    fib = [0] * (limit+2)
    fib[1] = 1
    for i in xrange(2, limit + 2):
        fib[i] = fib[i - 1] + fib[i -2]
    for i in xrange(limit):
        result[i] = fib[A[i]+1] & B[i]
    return result

# FibFrog
# Count the minimum number of jumps required for a frog to get to the other side of a river.
def fibonacciDynamic(n):
    # Generate and return all the Fibonacci numbers,
    # less than or equal to n, in descending order.
    # n must be larger than or equal to one.
    fib = [0] * (n + 2)
    fib[1] = 1
    for i in xrange(2, n+2):
        fib[i] = fib[i -1]+fib[i-2] 
        if fib[i] > n:
            return fib[i-1: 1: -1]
        elif fib[i] == n:
            return fib[i: 1: -1]
def solution(A):
    class Status(object):
        # Object to store the status of attempts
        __slots__ = ('position', 'moves')
        def __init__(self, pos, moves):
            self.position = pos
            self.moves = moves
            return
    lenA = len(A)
    fibonacci = fibonacciDynamic(lenA+1) # Fibonacci numbers
    statusQueue = [Status(-1,0)] # Initially we are at position -1 with 0 move.
    nextTry = 0 # We are not going to delete the tried attemp. 
                # So we need a pointer to the next attemp.
    accessed = [False] * len(A) # Were we in this position before?
    while True:
        if nextTry == len(statusQueue):
        # There is no unprocessed attemp. And we did not
        # find any path yet. So no path exists.
            return -1
    # Obtain the next attemp's status
    currentStatus = statusQueue[nextTry]
    nextTry += 1
    currentPos = currentStatus.position
    currentMoves = currentStatus.moves
    # Based upon the current attemp, we are trying any
    # possible move.
    for length in fibonacci:
        if currentPos + length == lenA:
            # Ohhh~ We are at the goal position!
            return currentMoves + 1
        elif currentPos + length > lenA \
            or A[currentPos + length] == 0 \
            or accessed[currentPos + length]:
            # Three conditions are moving too far, no leaf available for moving, 
            # and being here before respectively.
            # PAY ATTENTION: we are using Breadth First Search. 
            # If we were here before, the previous attemp must achieved here 
            # with less or same number of moves. 
            # With completely same future path, current attemp will never have 
            # less moves to goal than previous attemp. 
            # So it could be pruned.
            continue
        # Enqueue for later attemp.
        statusQueue.append(Status(currentPos + length, currentMoves +1))
        accessed[currentPos + length] = True

# MinMaxDivision
# Divide array A into K blocks and minimize the largest sum of any blocks.
def blocksNo(A, maxBlock):
    # Initially set the A[0] being an individual block
    blocksNumber = 1
    # The number of blocks, that A could be divided to with the restriction that, 
    # the sum of each block is less than or equal to maxBlock
    preBlockSum = A[0]
    for element in A[1:]:
        # Try to extend the previous block
        if preBlockSum + element > maxBlock:
            # Fail to extend the previous block, because
            # of the sum limitation maxBlock
            preBlockSum = element
            blocksNumber += 1
        else:
            preBlockSum += element
    return blocksNumber
def solution(K, A):
    blocksNeeded = 0 # Given the restriction on the sum of
                     # each block, how many blocks could
                     # the original A be divided to?
    resultLowerBound = max(A)
    resultUpperBound = sum(A)
    result = 0 # Minimal large sum
    # Handle two special cases
    if K == 1:      return resultUpperBound
    if K >= len(A): return resultLowerBound
    # Binary search the result
    while resultLowerBound <= resultUpperBound:
        resultMaxMid = (resultLowerBound + resultUpperBound) / 2
        blocksNeeded = blocksNo(A, resultMaxMid)
        if blocksNeeded <= K:
            # With large sum being resultMaxMid or resultMaxMid-,
            # we need blocksNeeded/blocksNeeded- blocks. While we
            # have some unused blocks (K - blocksNeeded), We could
            # try to use them to decrease the large sum.
            resultUpperBound = resultMaxMid - 1
            result = resultMaxMid
        else:
            # With large sum being resultMaxMid or resultMaxMid-,
            # we need to use more than K blocks. So resultMaxMid
            # is impossible to be our answer.
            resultLowerBound = resultMaxMid + 1
    return result

# NailingPlanks
# Count the minimum number of nails that allow a series of planks to be nailed.
def solution(A, B, C):
    result = -1         # Global result
    # Sort the planks according to firstly their begin position,
    # and then their end position
    planks = zip(A, B)
    planks.sort()
    # Sort the nails according to their position
    nails = sorted(enumerate(C), key = lambda x: x[1])
    nailsIndex = 0
    # Travel for each plank
    for plankIndex in xrange(len(planks)):
        plank = planks[plankIndex]
        # Find the first quanified nail in linear manner. Beware
        # that the planks are sorted. For any two adjacent planks,
        # the begin position of the latter one will be:
        #   either the same as the former's begin position
        #   or after the former's
        # In both cases, the nails, which before the nailsIndex
        # of the former plank's round, would never be candidates
        # in the latter plank's round. Thus we only need to search
        # nails from the previous nailsIndex position.
        while nailsIndex < len(nails):
            if nails[nailsIndex][1] < plank[0]:
                nailsIndex += 1
            elif nails[nailsIndex][1] > plank[1]:
                # And all the remaining nails > plank[1]
                # Impossible to find a quanlified nail
                return -1
            else:
                # plank[0] <= nails[nailsIndex][1] <= plank[1]
                break
        else:
            # Cannot find one
            return -1
        if plankIndex != 0 and plank[0] == planks[plankIndex-1][0]:
            # This plank and previous plank have the same begin
            # position. And the planks are sorted. So the end
            # position of this plank is after that of previous
            # plank. We continue the previous search.
            pass
        else:
            # This plank and previous plank have the different
            # begin position. We have to re-search from the
            # nailsIndex.
            tempRes = len(nails)  # Local result for this round
            tempIndex = nailsIndex
        # Find the first one in all the quanlified nails
        while tempIndex < len(nails) and \
            plank[0] <= nails[tempIndex][1] <= plank[1]:
            tempRes = min(tempRes, nails[tempIndex][0])
            tempIndex += 1
            # If we find a tempRes <= result, the final result
            # of current round will <= result. This tempRes
            # would never change the global result. Thus we
            # could ignore it, and continue the next round.
            if tempRes <= result:   break
        result = max(result, tempRes)
    return result+1

# AbsDistinct
# Compute number of distinct absolute values of sorted array elements.
def solution(A):
    abs_distinct =1
    current = max(abs(A[0]), abs(A[-1]))
    index_head = 0
    index_tail = len(A)-1
    while index_head <= index_tail:
        # We travel the array from the greatest
        # absolute value to the smallest.
        former = abs(A[index_head])
        if former == current:
            # Skip the heading elements, whose
            # absolute values are the same with
            # current recording one.
            index_head += 1
            continue
        latter = abs(A[index_tail])
        if latter == current:
            # Skip the tailing elements, whose
            # absolute values are the same with
            # current recording one.
            index_tail -= 1
            continue
        # At this point, both the former and
        # latter has different absolute value
        # from current recorded one.
        if former >= latter:
            # The next greatest value is former
            current = former
            index_head += 1
        else:
            # The next greatest value is latter
            current = latter
            index_tail -= 1
        # Meet with a new absolute value
        abs_distinct += 1
    return abs_distinct

# CountDistinctSlices
# Count the number of distinct slices (containing only unique numbers).
def solution(M, A):
    accessed = [-1] * (M + 1) # -1: not accessed before
                              # Non-negative: the previous occurrence position
    front, back = 0, 0
    result = 0
    for front in xrange(len(A)):
        if accessed[A[front]] == -1:
            # Met with a new unique item
            accessed[A[front]] = front
        else:
            # Met with a duplicate item
            # Compute the number of distinct slices between newBack-1 and back position.
            newBack = accessed[A[front]]+1
            result += (newBack-back)*(front-back+front-newBack+1)/2
            if result >= 1000000000:
                return 1000000000
            # Restore and set the accessed array
            for index in xrange(back, newBack):
                accessed[A[index]] = -1
            accessed[A[front]] = front
            back = newBack
    # Process the last slices
    result += (front-back+1)*(front-back+2)/2
    return min(result, 1000000000)

# MinAbsSumOfTwo
# Find the minimal absolute value of a sum of two elements.
def solution(A):
    A.sort()        # Sort A in non-decreasing order
    if A[0] >= 0:
        return A[0]+A[0] # All non-negative
    if A[-1] <= 0:
        return -A[-1]-A[-1] # All non-positive
    front, back = len(A)-1, 0
    minAbs = A[-1]+A[-1]                  # Final result
    # Travel the array from both ends to some center point.
    # See following post for the proof of this method.
    # http://codesays.com/2014/solution-to-min-abs-sum-of-two-by-codility
    while back <= front:
        temp = abs(A[back]+A[front])
        # Update the result if needed
        if temp < minAbs:
            minAbs = temp
        # Adjust the pointer for next trying
        if abs(A[back+1] + A[front]) <= temp:
            back += 1
        elif abs(A[back] + A[front-1]) <= temp:
            front -= 1
        else:
            back += 1; front -= 1
    return minAbs

# CountTriangles
# Count the number of triangles that can be built from a given set of edges.
def solution(A):
    n = len(A)
    result = 0
    A.sort()
    for first in xrange(n-2):
        third = first + 2
        for second in xrange(first+1, n-1):
            while third < n and A[first] + A[second] > A[third]:
                third += 1
            result += third-second-1
    return result

# MaxNonoverlappingSegments
# Find a maximal set of non((-))overlapping segments.
def solution(A, B):
    if len(A) < 1:
        return 0
    cnt = 1
    prev_end = B[0]
    for idx in xrange(1, len(A)):
        if A[idx] > prev_end: 
            cnt+=1
            prev_end = B[idx]
    return cnt

# TieRopes
# Tie adjacent ropes to achieve the maximum number of ropes of length >= K.
def solution(K, A):
    # The number of tied ropes, whose lengths
    # are greater than or equal to K.
    count = 0
    # The length of current rope (might be a tied one).
    length = 0
    for rope in A:
        length += rope  # Tied with the previous one.
        # Find a qualified rope. Prepare to find the next one.
        if length >= K:
            count += 1; length = 0
    return count

# NumberSolitaire
# In a given array, find the subset of maximal sum in which the distance between consecutive elements is at most 6.
NR_POSSIBLE_ROLLS = 6
MIN_VALUE = -10000000001
def solution(A):
    sub_solutions = [MIN_VALUE] * (len(A)+NR_POSSIBLE_ROLLS)
    sub_solutions[NR_POSSIBLE_ROLLS] = A[0]
    # iterate over all steps
    for idx in xrange(NR_POSSIBLE_ROLLS+1, len(A)+NR_POSSIBLE_ROLLS):
        max_previous = MIN_VALUE
        for previous_idx in xrange(NR_POSSIBLE_ROLLS):
            max_previous = max(max_previous, sub_solutions[idx-previous_idx-1])
        # the value for each iteration is the value at the A array
        # plus the best value from which this index can be reached
        sub_solutions[idx] = A[idx-NR_POSSIBLE_ROLLS] + max_previous
    return sub_solutions[len(A)+NR_POSSIBLE_ROLLS-1]

# MinAbsSum
# Given array of integers, find the lowest absolute sum of elements.
def slow_min_abs_sum(A):
    N = len(A)
    M = 0
    for i in xrange(N):
        A[i] = abs(A[i])
        M = max(A[i], M)
    S = sum(A)
    dp = [0]*(S+1)
    dp[0] = 1
    for j in xrange(N):
        for i in xrange(S, -1, -1):
            if (dp[i] == 1) and (i + A[j] <= S):
                dp[i + A[j]] = 1
    result = S 
    for i in xrange(S//2+1):
        if dp[i] == 1:
            result = min(result, S - 2 * i)
    return result

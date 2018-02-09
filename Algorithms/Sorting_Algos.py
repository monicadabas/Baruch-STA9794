import sys

"""
Selection sort

Input: unsorted array A

Algo: Find the minimum and swap it with element at 0 index
      Find 2nd minimum and swap it with element at 1 index
      Repeat till second last minimum

Time Complexity = O(n^2)
Space Complexity = O(1)

"""


def selection_sort(A):
    n = len(A)

    if n <= 1:
        return A

    # loop from index 0 till second last index n-2
    for i in range(n-1):
        index_min = i

        # find index of ith minimum element

        for j in range(i+1, n):
            if A[j] < A[index_min]:
                index_min = j

        # swap the ith element with ith minimum element

        A[i], A[index_min] = A[index_min], A[i]

    return A


"""
Bubble Sort

Input: unsorted array A

Algo: Compare the adjacent element starting from index 0 till second last element
      If the element at index i is greater than that on index i+1, swap them
      Repeat this n-1 times (n-1 passes)

      Optimization:
      since with each pass the lasrgest number will be at its sorted index,
      we dont need to compare all the element but only till the last unsorted element

      it may be possible that the array gets sorted before all the passes so we should
      be able to terminate it instead. Use flag for that

Time Complexity = O(n^2)
Space Complexity = O(1)

"""


def bubble_sort(A):
    n = len(A)

    if n <= 1:
        return A

    # with each pass the largest number will go to its right index
    for k in range(n-1):

        # if in a pass no swaps are made then array is sorted and we can finish
        flag = 0

        # we only need to compare till the last unsorted element hence n-1-k
        for i in range(n-1-k):
            if A[i] > A[i+1]:
                A[i], A[i+1] = A[i+1], A[i]
                flag = 1

        # if in a pass there are no swaps, array is sorted
        if flag == 0:
            break
    return A


"""
Insertion Sort

Input: unsorted array A

Algo: Array A is divided into a sorted and an unsorted section. Initially index 0 is in sorted array
      and array from index 1 till end is unsorted section.

      We start from element at index 1 till end (i), compare the element with the elements in sorted array.
      If the element at index i is smaller than the element in sorted array, then the element in sorted array moves
      one index to the right. Once we find an element in the sorted array that is less than the element at ith index
      or we reach index 0 in the sorted section, we insert the ith element in the sorted section.

Time Complexity = O(n^2) however it is better than bubble and selection sort as number of comparisons are less
Space Complexity = O(1)

"""


def insertion_sort(A):
    n = len(A)

    if n <= 1:
        return A

    for i in range(1, n):

        value = A[i]  # element at ith index to be inserted in sorted section
        hole = i  # index in sorted section for value

        # hole will keep shifting left until we find an element less than value or we reach index 0
        while hole > 0 and value < A[hole-1]:
            A[hole] = A[hole-1]
            hole -= 1

        # insert value at index hole
        A[hole] = value

    return A


"""
Merge Sort

Algo: It is a recursive algo. We start by dividing the array into two parts, left and right.
      We do the same for left and right arrays till the size of the parts is 1. if an array has only 1 element
      it is sorted. Hence, after dividing the array till it has only element, we start merging them.

      The merge algo takes first element from the sorted left and right arrays and fill the lower of those
      at the first index in final array and so on. Till the final arry has all the elements from its left and right
      sub arrays.

Time Complexity = O(n log n)
Space Complexity = O(n)

"""


# it merges the sorted L and R arrays into A (the final array) and returns it
def merge(L, R, A):
    i, j, k = 0, 0, 0  # current index of the L, R, A arrays
    nL, nR = len(L), len(R)

    while i < nL and j < nR:
        if L[i] < R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
        k += 1

    # if all the elements of one of the L or R arrays have been inserted, we need to insert the rest of the
    # elements in the other array to A

    while i < nL:
        A[k] = L[i]
        i += 1
        k += 1

    while j < nR:
        A[k] = R[j]
        j += 1
        k += 1

    return A


# the recursive part of the algo to divide the array until there is only 1 element in it
def merge_sort(A):
    n = len(A)

    if n < 2: return A

    mid = int(n/2)

    left = A[:mid]
    right = A[mid:]

    merge_sort(left)
    merge_sort(right)
    return merge(left, right, A)


"""
Quick Sort

Algo: It is a recursive algo and has two parts to the problem
      Part one: partition the array around pivot such that all the elements less than pivot
                are on the left and rest on the right of it.
                Repeat the process for left and right parts
      Part two: Partitioning to bring the pivot to its right index in the array
                partition index = 0, pivot is value at end index of the array
                For the values at index (i) from start till end -1:
                if the element is smaller than the pivot value, we swap the values at i and partition index
                and increment partition index by 1
                Once this has been done for all the elements till index end -1, we swap the value at pivot index
                and partition index. Finally return the partition index

      Optimization: To avoide the worst case time complexity of O(n^2) we can pick the pivot randomly,
                swap the value at pivot index with element at last index and continue with the above algo.

Time Complexity = O(n log n)
Space Complexity = O(1)

"""

import random


def partition(A, start, end):
    if start >= end:
        return

    # pick the pivot index randomly and swap the element there with the last element
    pivot_index = random.randint(start, end)
    A[pivot_index], A[end] = A[end], A[pivot_index]

    partition_index = start

    for i in range(start, end):
        if A[i] < A[end]:
            A[i], A[partition_index] = A[partition_index], A[i]
            partition_index += 1

    A[partition_index], A[end] = A[end], A[partition_index]

    return partition_index


# initially we only get A but need start and end. We cannot initialize end = 0 as position argument
# and apply conditions on it as it can be a passed argument later
# hence used *args

def quick_sort(A, *argv):
    if len(argv) == 0:
        start = 0
        end = len(A) -1  # end index of array A
    else:
        start = argv[0]
        end = argv[1]

    if start < end:
        partitionIndex = partition(A, start, end)
        quick_sort(A, start, partitionIndex-1)
        quick_sort(A, partitionIndex+1, end)


A = [4, 1, 6, 3, 9, 2]
quick_sort(A)
print(A)



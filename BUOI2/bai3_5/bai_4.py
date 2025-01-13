import random
import time

# Insertion Sort
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# Quick Sort
def quick_sort(arr):
    if len(arr) <= 10:  # Chuyển sang Insertion Sort nếu kích thước mảng nhỏ
        insertion_sort(arr)
    else:
        if len(arr) > 1:
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quick_sort(left) + middle + quick_sort(right)
    return arr

# Đo thời gian thực hiện
def measure_time(arr):
    start_time = time.time()
    sorted_arr = quick_sort(arr)
    end_time = time.time()
    return end_time - start_time

# Thử nghiệm với các mảng ngẫu nhiên có kích thước khác nhau
array_sizes = [10, 100, 1000, 10000, 100000]

for size in array_sizes:
    test_array = [random.randint(0, 1000) for _ in range(size)]
    execution_time = measure_time(test_array)
    print(f"Time taken to sort array of size {size}: {execution_time:.5f} seconds")

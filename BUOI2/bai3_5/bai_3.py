import random
import time

# Quick Sort tiêu chuẩn
def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort(arr, low, high):
    if low < high:
        p = partition(arr, low, high)
        quick_sort(arr, low, p - 1)
        quick_sort(arr, p + 1, high)

# Randomized Quick Sort
def randomized_partition(arr, low, high):
    pivot_index = random.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    return partition(arr, low, high)

def randomized_quick_sort(arr, low, high):
    if low < high:
        p = randomized_partition(arr, low, high)
        randomized_quick_sort(arr, low, p - 1)
        randomized_quick_sort(arr, p + 1, high)

# Đo thời gian thực hiện
def measure_time(sort_function, arr):
    start_time = time.time()
    sort_function(arr, 0, len(arr) - 1)
    end_time = time.time()
    return end_time - start_time

# Hàm chính
def main():
    sizes = [5000, 10000, 20000]
    for size in sizes:
        print(f"\nMảng kích thước: {size}")
        arr = [random.randint(1, 10000) for _ in range(size)]
        
        # Quick Sort tiêu chuẩn
        arr_standard = arr.copy()
        standard_time = measure_time(quick_sort, arr_standard)
        print(f"Quick Sort tiêu chuẩn: {standard_time:.4f} giây")
        
        # Randomized Quick Sort
        arr_randomized = arr.copy()
        randomized_time = measure_time(randomized_quick_sort, arr_randomized)
        print(f"Randomized Quick Sort: {randomized_time:.4f} giây")

if __name__ == "__main__":
    main()

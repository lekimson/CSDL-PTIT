import random
import time

# Merge Sort
def merge(arr, left, mid, right):
    n1 = mid - left + 1
    n2 = right - mid
    L = arr[left:left + n1]
    R = arr[mid + 1:mid + 1 + n2]

    i = j = 0
    k = left
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)
        merge(arr, left, mid, right)

# Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# Đo thời gian thực hiện
def measure_time(sort_function, arr):
    start_time = time.time()
    sort_function(arr)
    end_time = time.time()
    return end_time - start_time

# Hàm chính
def main():
    sizes = [5000, 10000, 20000]
    for size in sizes:
        print(f"\nMảng kích thước: {size}")
        arr = [random.randint(1, 10000) for _ in range(size)]
        
        # Đo thời gian Merge Sort
        arr_merge = arr.copy()
        merge_time = measure_time(lambda x: merge_sort(x, 0, len(x) - 1), arr_merge)
        print(f"Merge Sort: {merge_time:.4f} giây")

        # Đo thời gian Bubble Sort
        arr_bubble = arr.copy()
        bubble_time = measure_time(bubble_sort, arr_bubble)
        print(f"Bubble Sort: {bubble_time:.4f} giây")

if __name__ == "__main__":
    main()

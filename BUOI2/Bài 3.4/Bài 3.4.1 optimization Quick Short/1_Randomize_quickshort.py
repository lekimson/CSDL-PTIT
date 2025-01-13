import random


# Hàm quicksort sử dụng pivot ngẫu nhiên
def randomized_quickshort(arr, low, high):
    pivot_index = random.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    return partitionFunc(arr, low, high)

# Hàm quicksort đệ quy
def quick_short_randomized(arr, low, high):
    if low < high:
        p = randomized_quickshort(arr, low, high)
        quick_short_randomized(arr, low, p - 1)
        quick_short_randomized(arr, p + 1, high)

def merge_short_hybrid(arr, low, high):
    if high - low < 10:
        insertion_sort(arr, low, high)
    elif low < high:
            mid = (low + high) // 2
            merge_short_hybrid(arr, low, mid)
            merge_short_hybrid(arr, mid + 1, high)
            merge(arr, low, mid, high)
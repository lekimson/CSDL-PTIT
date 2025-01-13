def linear_search(arr, target):
    for index, value in enumerate(arr):
        if value == target:
            return index
    return -1

arr = [4, 2, 5, 1, 3]
print(linear_search(arr, 5))  # Output: 2
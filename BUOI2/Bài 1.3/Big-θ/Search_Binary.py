
def Search_Binary (arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1 
    return -1

arr = list(range(1,17)) # giá trị của mảng từ 1 đến 16
print(Search_Binary(arr, 10)) # 9
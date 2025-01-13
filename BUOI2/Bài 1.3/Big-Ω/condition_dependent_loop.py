def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2  # Tính chỉ số giữa
        if arr[mid] == target:
            return mid  # Trả về vị trí nếu tìm thấy
        elif arr[mid] < target:
            left = mid + 1  # Tìm bên phải
        else:
            right = mid - 1  # Tìm bên trái
    return -1  # Trả về -1 nếu không tìm thấy

# Danh sách cần tìm kiếm
arr = list(range(1, 17))  # [1, 2, 3, ..., 16]

# Gọi hàm binary_search
print(binary_search(arr, 10))  # Output: 9

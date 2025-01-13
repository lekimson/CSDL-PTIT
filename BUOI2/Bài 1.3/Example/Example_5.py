#đây là chương trình python sử dụng thuật toán quick short để sắp xếp mảng
def partition(arr, low, high):
    pivot = arr[high]  # Chọn pivot là phần tử cuối
    i = low - 1  # Chỉ số cho phần tử nhỏ hơn
    for j in range(low, high):
        if arr[j] < pivot:  # So sánh với pivot
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # Đổi chỗ
    arr[i + 1], arr[high] = arr[high], arr[i + 1]  # Đổi chỗ pivot
    return i + 1

def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)  # Phân vùng
        quick_sort(arr, low, pi - 1)  # Sắp xếp phần bên trái
        quick_sort(arr, pi + 1, high)  # Sắp xếp phần bên phải

arr = [5, 2, 9, 1, 5, 6]
quick_sort(arr, 0, len(arr) - 1)
print("Kết quả Quick Sort:", arr)  # Output: [1, 2, 5, 5, 6, 9] 
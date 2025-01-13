
#đây là chương trình python sử dụng thuật toán Quick Sort để sắp xếp mảng

def partitionFunc(arr, low, high):
    i = low - 1 #chỉ số của phần tử nhỏ hơn pivot
    pivot = arr[high]#chọn pivot là phần tử cuối cùng
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i] #đổi chỗ 2 phần tử
    arr[i + 1], arr[high] = arr[high], arr[i + 1] #đổi pivot vào vị trí đúng
    return i + 1 

def quickSort(arr, low, high):
    if low < high:
        pi = partitionFunc(arr, low, high)
        quickSort(arr, low, pi - 1) #sắp xếp nửa trái
        quickSort(arr, pi + 1, high) #sắp xếp nửa phải
        
if __name__ == "__main__":
    arr = [5, 9, 1, 2,5, 6]
    quickSort(arr, 0, len(arr) - 1)
    print("Mảng sau khi sắp xếp là: ", arr)  # In mảng đã sắp xếp
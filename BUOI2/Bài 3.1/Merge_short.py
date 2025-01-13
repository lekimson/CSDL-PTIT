#đây là chương trình python thực hiện sắp xếp mảng bằng phương pháp Merge Sort
def Merge(arr, left, right, mid):
    n1 = mid - left + 1  # Độ dài của mảng L
    n2 = right - mid     # Độ dài của mảng R
    
    L = arr[left:mid + 1]  # Mảng con L
    R = arr[mid + 1:right + 1]  # Mảng con R
    
    i = 0  # Chỉ số của mảng L
    j = 0  # Chỉ số của mảng R
    k = left  # Chỉ số của mảng arr

    # Trộn 2 mảng con L và R vào arr
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # Sao chép phần còn lại của mảng L (nếu có)
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    # Sao chép phần còn lại của mảng R (nếu có)
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

# Hàm MergeSort thực hiện việc chia mảng và gọi Merge
def MergeSort(arr, left, right):
    if left < right:
        mid = (left + right) // 2  # Tính chỉ số giữa
        MergeSort(arr, left, mid)  # Sắp xếp nửa trái
        MergeSort(arr, mid + 1, right)  # Sắp xếp nửa phải
        Merge(arr, left, right, mid)  # Trộn 2 nửa đã sắp xếp

if __name__ == "__main__":
    arr = [5, 9, 1, 2, 6]
    MergeSort(arr, 0, len(arr) - 1)  # Gọi MergeSort để sắp xếp mảng
    print("Mảng sau khi sắp xếp là: ", arr)  # In mảng đã sắp xếp

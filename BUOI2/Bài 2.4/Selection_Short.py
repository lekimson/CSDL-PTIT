
# đây là chương trình python sử dụng thuật toán Selection Short để sắp xếp mảng arr

def Selection_Short (arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
        
if __name__ == "__main__":
    arr = [5, 2, 9, 1, 5, 6]
    Selection_Short(arr)
    print("Kết quả Selection Short:", arr)  # Output: [1, 2, 5, 5, 6, 9]
    
    
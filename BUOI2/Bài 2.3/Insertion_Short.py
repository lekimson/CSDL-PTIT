#đây là chương trình python dùng thuật toán Insertion Short để sắp xếp mảng arr

# Hàm thực hiện sắp xếp mảng theo thuật toán Insertion Sort
def insertion_short(arr):
    n = len(arr)  
    for i in range(1, n): 
        key = arr[i]  # Lấy giá trị của phần tử hiện tại làm "key"
        j = i - 1  # Bắt đầu so sánh với phần tử phía trước

        # Dịch chuyển các phần tử lớn hơn "key" sang bên phải
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        
        # Đặt "key" vào vị trí đúng
        arr[j + 1] = key

if __name__ == "__main__":
    # Khởi tạo mảng cần sắp xếp
    arr = [5, 2, 9, 1, 5, 6]

    # Gọi hàm sắp xếp
    insertion_short(arr)

    # In kết quả sau khi sắp xếp
    print("Kết quả Insertion Sort:", arr)  # Output: [1, 2, 5, 5, 6, 9]

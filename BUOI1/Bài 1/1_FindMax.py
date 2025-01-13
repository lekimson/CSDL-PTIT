# Hàm FindMax: Tìm giá trị lớn nhất trong một danh sách (list)
def FindMax(A):
    # Kiểm tra nếu danh sách rỗng
    if not A:
        print("Error: Empty array!")  # Thông báo lỗi
        return None  # Trả về None để biểu thị lỗi
    
    # Giả sử phần tử đầu tiên là giá trị lớn nhất
    max_val = A[0]

    # Lặp qua các phần tử còn lại để tìm giá trị lớn nhất
    for i in range(1, len(A)):
        if A[i] > max_val:  # Nếu phần tử hiện tại lớn hơn giá trị lớn nhất
            max_val = A[i]  # Cập nhật giá trị lớn nhất
    
    return max_val  # Trả về giá trị lớn nhất

# Hàm main để khởi chạy chương trình
def main():
    # Khởi tạo danh sách các số nguyên
    A = [3, 5, 2, 9, 1]
    
    # Gọi hàm FindMax để tìm giá trị lớn nhất trong danh sách
    max_val = FindMax(A)
    
    # Nếu kết quả không phải None, in ra giá trị lớn nhất
    if max_val is not None:
        print(f"Giá trị lớn nhất trong mảng là: {max_val}")

# Điểm khởi đầu của chương trình
if __name__ == "__main__":
    main()

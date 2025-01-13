def LinearSearch (A, x):
    for i in range(len(A)):  # lệnh này lặp từng giá trị trong mảng
        if A[i] == x:
            return i # trả về giá trị nếu tìm thấy
    return -1 # không tìm thấy 
# hàm chính của chương trình
def main():
    A = [3, 5, 2, 9, 1]
    x = 9
    result = LinearSearch(A, x) #gán giá trị cho result bằng cách gọi hàm và truyền tham số cho nó
    if result != -1:
        print(f"Phần tử {x} được tìm thấy tại vị trí: {result}")
    else:
        print(f"Phần tử {x} không tồn tại trong mảng")

# chạy chương trình chính
if __name__ == "__main__":
    main()

    
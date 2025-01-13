def InsertionSort(A):
    # Lặp qua từng phần tử của mảng, bắt đầu từ phần tử thứ hai
    for i in range(1, len(A)):
        key = A[i]  # Lưu giá trị hiện tại để so sánh
        j = i - 1   # Vị trí phần tử đứng trước

        # Dịch chuyển các phần tử lớn hơn 'key' sang phải
        while j >= 0 and A[j] > key:
            A[j + 1] = A[j]  # Dịch chuyển phần tử
            j -= 1           # Lùi về vị trí trước đó

        # Chèn 'key' vào vị trí đúng
        A[j + 1] = key

# Hàm chính để chạy chương trình
def main():
    A = [5, 2, 9, 1, 5, 6]  # Mảng đầu vào
    InsertionSort(A)         # Gọi hàm sắp xếp chèn
    print("Mảng sau khi sắp xếp:", A)  # In kết quả

# Điểm bắt đầu của chương trình
if __name__ == "__main__":
    main()

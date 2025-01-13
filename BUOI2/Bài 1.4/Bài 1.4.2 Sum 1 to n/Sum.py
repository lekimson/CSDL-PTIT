# Chương trình tính tổng từ 1 đến n
# Hàm tính toán 1 đến n
def sum_1_to_n(n):
    s = 0
    for i in range(1, n + 1):
        s += i
    return s  # Trả về kết quả

if __name__ == "__main__":
    n = int(input("Nhập n: "))  # Input
    result = sum_1_to_n(n)
    print("Tổng từ 1 đến", n, "là", result)  # Output

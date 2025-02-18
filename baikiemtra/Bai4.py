#đây là chương trình về sự khác biệt giữa 2 chuỗi
def levenshtein_distance(s, t):
    m, n = len(s), len(t)
    
    # Tạo bảng dp có kích thước (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Khởi tạo hàng và cột đầu tiên
    for i in range(m + 1):
        dp[i][0] = i  # Xóa tất cả các ký tự của s để thành rỗng
    for j in range(n + 1):
        dp[0][j] = j  # Thêm tất cả các ký tự của t vào chuỗi rỗng
    
    # Điền bảng dp theo quy tắc Levenshtein
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:  # Ký tự giống nhau => không tốn chi phí
                dp[i][j] = dp[i - 1][j - 1]
            else:  # Chọn phép biến đổi tối ưu
                dp[i][j] = min(
                    dp[i - 1][j] + 1,   # Xóa ký tự
                    dp[i][j - 1] + 1,   # Thêm ký tự
                    dp[i - 1][j - 1] + 1  # Thay thế ký tự
                )
    
    return dp[m][n]  # Khoảng cách Levenshtein giữa s và t

# Kiểm tra với ví dụ
s = "Le Van Tam"
t = "Tam Le"
print(levenshtein_distance(s, t))  # Output: 3

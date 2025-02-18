from collections import deque

def max_sliding_window(num_list, k):
    if not num_list or k == 0:
        return []
    
    dq = deque()  # Lưu chỉ số của các phần tử trong cửa sổ
    res = []

    for i in range(len(num_list)):
        # Loại bỏ phần tử không còn trong cửa sổ
        if dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Loại bỏ các phần tử nhỏ hơn phần tử hiện tại
        while dq and num_list[dq[-1]] < num_list[i]:
            dq.pop()
        
        # Thêm chỉ số phần tử hiện tại vào deque
        dq.append(i)
        
        # Lấy phần tử lớn nhất của cửa sổ
        if i >= k - 1:
            res.append(num_list[dq[0]])

    return res

# Test
num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
k = 3
print(max_sliding_window(num_list, k))

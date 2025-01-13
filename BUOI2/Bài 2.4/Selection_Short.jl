# Đây là hàm thực hiện thuật toán Selection Sort để sắp xếp mảng
function Selection_Short!(arr)
    n = length(arr)  # Lấy độ dài của mảng
    for i in 1:n-1  # Duyệt qua từng phần tử trong mảng (lặp từ 1 đến n-1)
        min_index = i  # Giả sử phần tử tại vị trí i là nhỏ nhất

        # Tìm phần tử nhỏ nhất trong đoạn còn lại của mảng
        for j in i+1:n
            if arr[j] < arr[min_index]  # So sánh để tìm chỉ số phần tử nhỏ nhất
                min_index = j  # Cập nhật chỉ số của phần tử nhỏ nhất
            end
        end

        # Hoán đổi phần tử nhỏ nhất với phần tử tại vị trí i
        arr[i], arr[min_index] = arr[min_index], arr[i]
    end
end

# Phần chính của chương trình
arr = [5, 2, 9, 1, 5, 6]  # Khởi tạo mảng cần sắp xếp

# Gọi hàm Selection Sort để sắp xếp mảng
Selection_Short!(arr)

# In kết quả sau khi sắp xếp
println("Kết quả Selection Sort: ", arr)  # Output: [1, 2, 5, 5, 6, 9]

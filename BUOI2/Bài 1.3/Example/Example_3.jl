function Binary_Search(arr, target)
    # Hàm tìm kiếm nhị phân
    left, right = 0, length(arr) 
    
    while left <= right
        # Tính chỉ số giữa (mid)
        mid = div(left + right, 2)  # div để chia lấy phần nguyên
        
        # Kiểm tra nếu phần tử tại chỉ số giữa bằng với giá trị cần tìm
        if arr[mid] == target
            return mid  # Trả về chỉ số nếu tìm thấy
        
        # Nếu giá trị ở giữa nhỏ hơn giá trị cần tìm, tìm ở nửa phải
        elseif arr[mid] < target
            left = mid + 1
        
        # Nếu giá trị ở giữa lớn hơn giá trị cần tìm, tìm ở nửa trái
        else
            right = mid - 1
        end
    end
    
    # Nếu không tìm thấy giá trị, trả về -1
    return -1
end


arr = [1, 3, 5, 7, 9, 11]
# Tìm kiếm giá trị 7 trong mảng và in kết quả
println(Binary_Search(arr, 7))  # output = 4

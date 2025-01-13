function FindMax(A::Vector{Int})
    # Kiểm tra nếu mảng rỗng
    if isempty(A)
        println("Error: Empty array!")
        return nothing
    end
    
    # Giả sử giá trị lớn nhất là phần tử đầu tiên
    max_val = A[1]
    
    # Duyệt qua các phần tử còn lại để tìm giá trị lớn nhất
    for i in 2:length(A)
        if A[i] > max_val
            max_val = A[i]
        end
    end
    
    # Trả về giá trị lớn nhất
    return max_val
end

function main()
    # Khởi tạo mảng đầu vào
    A = [3, 5, 2, 9, 1]
    
    # Tìm giá trị lớn nhất trong mảng
    max_val = FindMax(A)
    
    # In giá trị lớn nhất nếu mảng không rỗng
    if max_val !== nothing
        println("Giá trị lớn nhất trong mảng là: ", max_val)
    end
end

# Chạy chương trình
main()

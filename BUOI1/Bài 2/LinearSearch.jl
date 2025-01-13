
function LinearSearch(A::Vector{Int}, x::Int)
    for i in 1:length(A) # Lặp qua từng phần tử trong mảng
        if A[i] == x  # Kiểm tra nếu phần tử tại vị trí i bằng x
            return i - 1  # To match 0-based indexing if needed
        end
    end
    return -1  # không tìm thấy
end

#hàm chính của chương trình
function main()
      # Khởi tạo mảng A và giá trị cần tìm x
    A = [3, 5, 2, 9, 1]
    x = 9
    index = LinearSearch(A, x) # gọi hàm và truyền giá trị của mảng A và giá trị cần tìm x
    if index != -1
        println("Giá trị $x được tìm thấy tại vị trí: $index")
    else
        println("Giá trị $x không tồn tại trong mảng.")
    end
end

main()

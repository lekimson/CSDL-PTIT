
function InsertionSort!(A::Vector{Int})
    # Lặp qua từng phần tử trong mảng, bắt đầu từ phần tử thứ hai
    for i in 2:length(A)
        key = A[i]   # Lưu giá trị hiện tại (key)
        j = i - 1    # Vị trí phần tử đứng trước

        # Dịch chuyển các phần tử lớn hơn 'key' sang phải
        while j >= 1 && A[j] > key
            A[j + 1] = A[j]  # Dịch chuyển phần tử
            j -= 1           # Lùi về vị trí trước đó
        end

        # Chèn 'key' vào vị trí đúng
        A[j + 1] = key
    end

    return A  # Trả về mảng đã sắp xếp
end

# Hàm chính để chạy chương trình
function main()
    A = [5, 2, 9, 1, 5, 6]  # Khởi tạo mảng ban đầu
    InsertionSort!(A)        # Gọi hàm sắp xếp chèn
    println("Mảng sau khi sắp xếp: ", A)  # In kết quả
end

main()

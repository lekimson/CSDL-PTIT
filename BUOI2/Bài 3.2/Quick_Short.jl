# Đây là chương trình Julia sử dụng thuật toán Quick Sort để sắp xếp mảng
function partitionFunc(arr, low::Int , high::Int)
    pivot = arr[high]  # Chọn phần tử cuối cùng làm pivot
    i = low - 1  # Khởi tạo chỉ số cho phần tử nhỏ hơn pivot
    for j in low:high-1  # Duyệt qua các phần tử trong mảng
        if arr[j] < pivot  # Nếu phần tử nhỏ hơn pivot
            i += 1  # Tăng chỉ số i
            arr[i], arr[j] = arr[j], arr[i]  # Hoán đổi phần tử
        end
    end
    arr[i+1], arr[high] = arr[high], arr[i+1]  # Đưa pivot vào vị trí đúng
    return i+1  # Trả về chỉ số của pivot
end

function quickSort!(arr, low::Int, high::Int)
    if low < high  # Nếu mảng có ít nhất 2 phần tử
        pi = partitionFunc(arr, low, high)  # Phân vùng mảng
        quickSort!(arr, low, pi-1)  # Sắp xếp phần mảng bên trái của pivot
        quickSort!(arr, pi+1, high)  # Sắp xếp phần mảng bên phải của pivot
    end
    return arr  # Trả về mảng đã sắp xếp
end

# Kiểm tra chương trình
arr = [5, 2, 9, 1, 5, 6]
quickSort!(arr, 1, length(arr))  # Đánh số từ 1, gọi quickSort
println("Kết quả Quick Sort: ", arr)  # Output: [1, 2, 5, 5, 6, 9]

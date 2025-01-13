# Đây là chương trình JUlia để sắp xếp mảng sử dụng thuật toán Merge Sort

# Hàm Merge thực hiện việc trộn 2 mảng con đã được sắp xếp vào mảng chính
function Merge(arr, left::Int, right::Int, mid::Int)
    # Tính độ dài của các mảng con L và R
    n1 = mid - left + 1
    n2 = right - mid
    
    # Tạo các mảng tạm L và R để chứa các phần tử từ mảng arr
    L = Array{eltype(arr)}(undef, n1)
    R = Array{eltype(arr)}(undef, n2)

    # Sao chép dữ liệu từ mảng arr vào các mảng tạm L và R
    for i in 1:n1
        L[i] = arr[left + i - 1]
    end
    for j in 1:n2
        R[j] = arr[mid + j]
    end

    # Khởi tạo các biến chỉ mục
    i, j = 1, 1
    k = left

    # Trộn các phần tử từ mảng L và R vào mảng arr
    while (i <= n1) && (j <= n2)
        if L[i] <= R[j]  # Nếu phần tử trong L nhỏ hơn hoặc bằng phần tử trong R
            arr[k] = L[i]  # Đưa phần tử từ L vào arr
            i += 1
        else
            arr[k] = R[j]  # Đưa phần tử từ R vào arr
            j += 1
        end
        k += 1  # Tăng chỉ số k để tiếp tục cập nhật mảng arr
    end

    # Sao chép phần còn lại của mảng L nếu có
    while i <= n1
        arr[k] = L[i]
        i += 1
        k += 1
    end

    # Sao chép phần còn lại của mảng R nếu có
    while j <= n2
        arr[k] = R[j]
        j += 1
        k += 1
    end
end

# Hàm Merge_Short! thực hiện thuật toán Merge Sort
function Merge_Short!(arr, left::Int, right::Int)
    if left < right  # Nếu mảng có ít nhất 2 phần tử
        # Tính chỉ số giữa của mảng
        mid = div(right + left, 2)
        
        # Đệ quy chia mảng thành 2 phần
        Merge_Short!(arr, left, mid)
        Merge_Short!(arr, mid + 1, right)
        
        # Sau khi chia xong, thực hiện trộn lại 2 mảng con đã sắp xếp
        Merge(arr, left, right, mid)
    end
    return arr  # Trả về mảng đã sắp xếp
end

# Kiểm tra chương trình
using Random
Random.seed!(0) # Cố định giá trị random
arr = [5, 2, 9, 1, 5, 6]

# Gọi hàm Merge_Short! để sắp xếp mảng
Merge_Short!(arr, 1, length(arr)) 

# In kết quả mảng sau khi sắp xếp
println("Mảng sau khi sắp xếp: ", arr)

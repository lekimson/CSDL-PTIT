#đây là chương trình julia về sắp xếp mảng theo thuật toán Bubble Short
function bubbleSort(arr)
    n = length(arr)
    for i in 1:n
        for j in 1:n-i
            if arr[j] > arr[j+1]
                arr[j], arr[j+1] = arr[j+1], arr[j]
            end
        end
    end
    return arr
end
arr = [5,9,1,6,8]
bubbleSort(arr)
println("Kết quả sau khi sắp xếp: ",arr)
# đây là chương trình julia về tính tổng 1 đến n
function sum_1_to_n(n)
    s = 0 
    for i in 1:n
        s += i
    end
    return s
end

println("Nhap n: ")
n = parse(Int64, readline())
result = sum_1_to_n(n)
println("Tong tu 1 den n la: ", result)
# Module nhập liệu
function InputNumber()
    while true
        print("Nhập vào một số nguyên không âm n: ")
        input = readline()
        n = tryparse(Int, input)
        if isnothing(n)
            println("Error: Vui lòng nhập một số nguyên hợp lệ.")
        elseif n < 0
            println("Error: n phải là số nguyên không âm. Vui lòng thử lại.")
        else
            return n
        end
    end
end

# Module tính toán
function ComputeFactorial(n::Int)
    if n == 0
        return 1
    else
        return n * ComputeFactorial(n - 1)
    end
end

# Module xuất kết quả
function OutputResult(result::Int)
    println("Giai thừa của n là: $result")
end

# Điều phối toàn bộ
function Factorial()
    n = InputNumber()
    result = ComputeFactorial(n)
    OutputResult(result)
end

function main()
    Factorial()
end

main()

def InputNumber():
    while True:
        try:
            n = int(input("Nhập vào một số nguyên không âm n: "))
            if n < 0:
                print("Error: n phải là số nguyên không âm. Vui lòng thử lại.")
                continue
            return n
        except ValueError:
            print("Error: Vui lòng nhập một số nguyên hợp lệ.")

# Module tính toán
def ComputeFactorial(n):
    if n == 0:
        return 1
    else:
        return n * ComputeFactorial(n - 1)

# Module xuất kết quả
def OutputResult(result):
    print(f"Giai thừa của n là: {result}")

# Điều phối toàn bộ
def Factorial():
    n = InputNumber()
    result = ComputeFactorial(n)
    OutputResult(result)

def main():
    Factorial()

if __name__ == "__main__":
    main()

//đây là chương trình tính tổng từ 1 đến n
#include <iostream>
using namespace std;

int sum_1_to_n(int n)
{
    int sum = 0; // thêm dấu chấm phẩy
    for (int i = 1; i <= n; i++)
    { // sửa điều kiện vòng lặp
        sum += i;
    }
    return sum; // trả về kết quả
}

int main()
{
    int n = 5;
    cout << "The total of 1 to " << n << " is " << sum_1_to_n(n) << endl;
}

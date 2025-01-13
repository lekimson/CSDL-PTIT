// Sum 1 to n
#include <iostream>
using namespace std;

//hàm tính tổng từ 1 đến n
int sum_1_to_n(int n)
{
    int s = 0;
    for (int i = 1; i <= n; i++)
    {
        s += i;
    }
    return s;
}
int main()
{

    cout << " nhap n: ";
    int n;
    cin >> n;
    //gọi hàm sum_1_to_n và in kết quả
    int result = sum_1_to_n(n);
    cout << "Sum = " << result << endl;
    return 0;
}
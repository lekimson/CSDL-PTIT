#include <iostream>

using namespace std;

void PrintNumber ( int n)
{
    // vòng lặp for để đếm số từ 
    for (int i = 1; i <= n; i++)
    {
        cout << i << " " ;
    }
    //xuống hàng khi xong vòng lặp
    cout<< endl;
}

int main ()
{
    //gọi hàm
    PrintNumber(5);
    return 0;
}

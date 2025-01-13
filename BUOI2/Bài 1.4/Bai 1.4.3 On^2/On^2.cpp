//đây là chương trình c++ về on^2
#include <iostream>
using namespace std;

int main()
{
    // Nhập giá trị n 
    cout << "n = "; 
    int n; 
    cin >> n;

    // Biến count để đếm số bước thực hiện
    int count = 0;

    // Vòng lặp bên ngoài chạy từ 0 đến n-1
    for (int i = 0; i < n; i++) 
    {
        // Vòng lặp bên trong chạy từ 0 đến n-1
        for (int j = 0; j < n; j++) 
        {
            // Tăng biến count mỗi lần vòng lặp bên trong chạy
            count++; 
        }
    }

    // Xuất số bước thực hiện (count) ra màn hình
    cout << "So buoc thuc hien = " << count << endl; 
    return 0; 
}

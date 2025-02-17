#include <iostream>
using namespace std;

// Hàm kiểm tra một số có các chữ số khác nhau và <= 5
bool isValid(int num) {
    bool used[6] = {false}; // Chỉ cần kiểm tra chữ số từ 0 đến 5
    while (num > 0) {
        int digit = num % 10;
        if (digit > 5 || used[digit]) {
            return false;
        }
        used[digit] = true;
        num /= 10;
    }
    return true;
}

// Hàm đếm số hợp lệ trong khoảng [L, R]
int countValidNumbers(int L, int R) {
    int count = 0;
    for (int i = L; i <= R; ++i) {
        if (isValid(i)) {
            count++;
        }
    }
    return count;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int L, R;
        cin >> L >> R;
        cout << countValidNumbers(L, R) << endl;
    }
    return 0;
}

#include <iostream>
#include <queue>
using namespace std;

string findSmallestMultiple(int N) {
    queue<string> q;
    q.push("9");
    
    while (!q.empty()) {
        string num = q.front();
        q.pop();
        
        // Chuyển số từ chuỗi sang số nguyên lớn
        long long value = stoll(num);
        if (value % N == 0) return num;
        
        q.push(num + "0");
        q.push(num + "9");
    }
    
    return "-1"; // Trường hợp không thể xảy ra
}

int main() {
    int T, N;
    cin >> T;
    while (T--) {
        cin >> N;
        cout << findSmallestMultiple(N) << endl;
    }
    return 0;
}

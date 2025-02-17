#include <iostream>
#include <queue>
using namespace std;

// Hàm đếm số BDN nhỏ hơn N
long long countBDN(long long N) {
    queue<long long> q;
    q.push(1);
    long long count = 0;
    
    while (!q.empty()) {
        long long num = q.front();
        q.pop();
        
        if (num <= N) {
            count++;
            q.push(num * 10);   // Thêm số có dạng "num0"
            q.push(num * 10 + 1); // Thêm số có dạng "num1"
        }
    }
    
    return count;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        long long N;
        cin >> N;
        cout << countBDN(N) << endl;
    }
    return 0;
}

#include <iostream>
#include <queue>
#include <unordered_set>
using namespace std;

// Hàm tìm số BDN nhỏ nhất chia hết cho N
long long findSmallestBDN(int N) {
    queue<long long> q;
    unordered_set<long long> visited;
    q.push(1);
    
    while (!q.empty()) {
        long long num = q.front();
        q.pop();
        
        if (num % N == 0) {
            return num;
        }
        
        if (visited.find(num) == visited.end()) {
            visited.insert(num);
            q.push(num * 10);   // Thêm số có dạng "num0"
            q.push(num * 10 + 1); // Thêm số có dạng "num1"
        }
    }
    return -1; // Không thể xảy ra
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int N;
        cin >> N;
        cout << findSmallestBDN(N) << endl;
    }
    return 0;
}

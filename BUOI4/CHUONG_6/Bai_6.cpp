#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>
using namespace std;

// Hàm tìm số bước ít nhất để biến N thành 1
int minOperations(int N) {
    queue<pair<int, int>> q;
    unordered_set<int> visited;
    q.push({N, 0}); // (giá trị hiện tại, số bước)
    
    while (!q.empty()) {
        auto [current, steps] = q.front();
        q.pop();
        
        if (current == 1) {
            return steps;
        }
        
        if (visited.find(current) == visited.end()) {
            visited.insert(current);
            q.push({current - 1, steps + 1}); // Thao tác trừ 1
            
            // Thử tất cả các cặp (u, v) sao cho u * v = current
            for (int i = 2; i * i <= current; ++i) {
                if (current % i == 0) {
                    q.push({max(i, current / i), steps + 1}); // Thao tác max(u, v)
                }
            }
        }
    }
    return -1;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int N;
        cin >> N;
        cout << minOperations(N) << endl;
    }
    return 0;
}
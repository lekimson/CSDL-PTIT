#include <iostream>
#include <queue>
#include <unordered_set>
using namespace std;

// Hàm tìm số bước ít nhất để biến S thành T
int minOperations(int S, int T) {
    queue<pair<int, int>> q;
    unordered_set<int> visited;
    q.push({S, 0}); // (giá trị hiện tại, số bước)
    
    while (!q.empty()) {
        auto [current, steps] = q.front();
        q.pop();
        
        if (current == T) {
            return steps;
        }
        
        if (visited.find(current) == visited.end() && current > 0) {
            visited.insert(current);
            q.push({current * 2, steps + 1}); // Thao tác nhân 2
            q.push({current - 1, steps + 1}); // Thao tác trừ 1
        }
    }
    return -1;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int S, T;
        cin >> S >> T;
        cout << minOperations(S, T) << endl;
    }
    return 0;
}
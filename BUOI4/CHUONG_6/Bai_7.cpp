#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>
using namespace std;

// Kiểm tra xem hai từ có khác nhau đúng 1 ký tự hay không
bool isOneLetterDiff(const string &a, const string &b) {
    int count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            count++;
            if (count > 1) return false;
        }
    }
    return count == 1;
}

// Hàm tìm khoảng cách ngắn nhất giữa s và t trong danh sách từ S
int shortestWordPath(const vector<string> &wordList, const string &s, const string &t) {
    unordered_set<string> words(wordList.begin(), wordList.end());
    if (words.find(t) == words.end()) return -1; // Nếu từ t không có trong danh sách
    
    queue<pair<string, int>> q;
    q.push({s, 1}); // (từ hiện tại, số bước)
    words.erase(s);
    
    while (!q.empty()) {
        auto [current, steps] = q.front();
        q.pop();
        
        if (current == t) {
            return steps;
        }
        
        for (auto it = words.begin(); it != words.end();) {
            if (isOneLetterDiff(current, *it)) {
                q.push({*it, steps + 1});
                it = words.erase(it); // Xóa từ đã thăm khỏi danh sách
            } else {
                ++it;
            }
        }
    }
    return -1;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int n;
        string s, t;
        cin >> n >> s >> t;
        vector<string> wordList(n);
        for (int i = 0; i < n; ++i) {
            cin >> wordList[i];
        }
        cout << shortestWordPath(wordList, s, t) << endl;
    }
    return 0;
}

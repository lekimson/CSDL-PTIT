#include <iostream>
#include <stack>
#include <vector>
using namespace std;

int maxValidParentheses(string s) {
    int n = s.length();
    vector<bool> valid(n, false);  // Đánh dấu ký tự thuộc cặp ngoặc đúng
    stack<int> st;

    // Duyệt chuỗi để đánh dấu các ký tự thuộc cặp đúng
    for (int i = 0; i < n; i++) {
        if (s[i] == '(') {
            st.push(i);
        } else if (!st.empty()) {
            valid[i] = valid[st.top()] = true;
            st.pop();
        }
    }

    // Tính tổng độ dài của các đoạn ngoặc đúng
    int maxLength = 0, currentLength = 0;
    for (int i = 0; i < n; i++) {
        if (valid[i]) {
            currentLength++;
        } else {
            maxLength += (currentLength >= 2) ? currentLength : 0;
            currentLength = 0;
        }
    }
    maxLength += (currentLength >= 2) ? currentLength : 0;

    return maxLength;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        string P;
        cin >> P;
        cout << maxValidParentheses(P) << endl;
    }
    return 0;
}

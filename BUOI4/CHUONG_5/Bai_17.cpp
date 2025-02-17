#include <iostream>
#include <stack>
#include <string>
using namespace std;

int longestValidParentheses(string s) {
    stack<int> st;
    int maxLength = 0;
    st.push(-1);
    
    for (int i = 0; i < s.length(); i++) {
        if (s[i] == '(') {
            st.push(i);
        } else {
            st.pop();
            if (!st.empty()) {
                maxLength = max(maxLength, i - st.top());
            } else {
                st.push(i);
            }
        }
    }
    return maxLength;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        string S;
        cin >> S;
        cout << longestValidParentheses(S) << endl;
    }
    return 0;
}

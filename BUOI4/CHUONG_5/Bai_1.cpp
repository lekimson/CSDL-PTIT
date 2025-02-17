#include <iostream>
#include <stack>
#include <string>
using namespace std;

bool hasRedundantParentheses(const string& expression) {
    stack<char> s;
    for (char ch : expression) {
        if (ch == ')') {
            bool hasOperator = false;
            int elementCount = 0;
            while (!s.empty() && s.top() != '(') {
                char top = s.top();
                s.pop();
                if (top == '+' || top == '-' || top == '*' || top == '/') {
                    hasOperator = true;
                }
                elementCount++;
            }
            if (!s.empty()) s.pop(); // Pop the opening bracket
            if (!hasOperator || elementCount == 1) return true;
        } else {
            s.push(ch);
        }
    }
    return false;
}

int main() {
    int T;
    cin >> T;
    cin.ignore();
    while (T--) {
        string exp;
        getline(cin, exp);
        cout << (hasRedundantParentheses(exp) ? "Yes" : "No") << endl;
    }
    return 0;
}
#include <iostream>
#include <stack>
#include <string>
#include <cctype>
using namespace std;

// Hàm tính toán giá trị của biểu thức trung tố
int precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/') return 2;
    return 0;
}

int applyOperation(int a, int b, char op) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return a / b;
    }
    return 0;
}

int evaluateInfix(string s) {
    stack<int> values;
    stack<char> ops;
    
    for (int i = 0; i < s.length(); i++) {
        if (isspace(s[i])) continue;
        
        if (isdigit(s[i])) {
            int val = 0;
            while (i < s.length() && isdigit(s[i])) {
                val = val * 10 + (s[i] - '0');
                i++;
            }
            values.push(val);
            i--;
        } else if (s[i] == '(') {
            ops.push(s[i]);
        } else if (s[i] == ')') {
            while (!ops.empty() && ops.top() != '(') {
                int val2 = values.top(); values.pop();
                int val1 = values.top(); values.pop();
                char op = ops.top(); ops.pop();
                values.push(applyOperation(val1, val2, op));
            }
            ops.pop();
        } else {
            while (!ops.empty() && precedence(ops.top()) >= precedence(s[i])) {
                int val2 = values.top(); values.pop();
                int val1 = values.top(); values.pop();
                char op = ops.top(); ops.pop();
                values.push(applyOperation(val1, val2, op));
            }
            ops.push(s[i]);
        }
    }
    
    while (!ops.empty()) {
        int val2 = values.top(); values.pop();
        int val1 = values.top(); values.pop();
        char op = ops.top(); ops.pop();
        values.push(applyOperation(val1, val2, op));
    }
    
    return values.top();
}

int main() {
    int T;
    cin >> T;
    cin.ignore();
    while (T--) {
        string exp;
        getline(cin, exp);
        cout << evaluateInfix(exp) << endl;
    }
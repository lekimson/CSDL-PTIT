#include <iostream>
#include <stack>
using namespace std;

// Hàm xác định độ ưu tiên của toán tử
int precedence(char op) {
    if (op == '^') return 3;
    if (op == '*' || op == '/') return 2;
    if (op == '+' || op == '-') return 1;
    return 0;
}

// Hàm chuyển đổi biểu thức trung tố sang hậu tố
string infixToPostfix(string exp) {
    stack<char> s;
    string result;
    
    for (char c : exp) {
        if (isalnum(c)) {  // Nếu là toán hạng (chữ hoặc số)
            result += c;
        } else if (c == '(') {
            s.push(c);
        } else if (c == ')') {
            while (!s.empty() && s.top() != '(') {
                result += s.top();
                s.pop();
            }
            s.pop();  // Bỏ dấu '('
        } else {  // Nếu là toán tử
            while (!s.empty() && precedence(s.top()) >= precedence(c)) {
                result += s.top();
                s.pop();
            }
            s.push(c);
        }
    }
    
    // Đưa các toán tử còn lại vào kết quả
    while (!s.empty()) {
        result += s.top();
        s.pop();
    }
    
    return result;
}

int main() {
    int T;
    cin >> T;
    cin.ignore();
    while (T--) {
        string exp;
        getline(cin, exp);
        cout << infixToPostfix(exp) << endl;
    }
    return 0;
}

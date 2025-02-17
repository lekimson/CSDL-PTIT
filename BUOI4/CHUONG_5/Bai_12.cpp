#include <iostream>
#include <stack>
#include <string>
using namespace std;

// Hàm tính toán giá trị của biểu thức tiền tố
int evaluatePrefix(string exp) {
    stack<int> s;
    int n = exp.length();
    
    for (int i = n - 1; i >= 0; i--) {
        char ch = exp[i];
        
        if (isdigit(ch)) {
            s.push(ch - '0');
        } else {
            int op1 = s.top(); s.pop();
            int op2 = s.top(); s.pop();
            
            switch (ch) {
                case '+': s.push(op1 + op2); break;
                case '-': s.push(op1 - op2); break;
                case '*': s.push(op1 * op2); break;
                case '/': s.push(op1 / op2); break;
            }
        }
    }
    
    return s.top();
}

int main() {
    int T;
    cin >> T;
    cin.ignore();
    while (T--) {
        string exp;
        getline(cin, exp);
        cout << evaluatePrefix(exp) << endl;
    }
    return 0;
}

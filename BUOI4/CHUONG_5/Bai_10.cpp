#include <iostream>
#include <stack>
#include <string>
using namespace std;

// Chuyển đổi biểu thức hậu tố sang trung tố
string postfixToInfix(string exp) {
    stack<string> s;
    int n = exp.length();
    
    for (int i = 0; i < n; i++) {
        char ch = exp[i];
        
        if (isalnum(ch)) {
            s.push(string(1, ch));
        } else {
            string op2 = s.top(); s.pop();
            string op1 = s.top(); s.pop();
            string newExp = "(" + op1 + ch + op2 + ")";
            s.push(newExp);
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
        cout << postfixToInfix(exp) << endl;
    }
    return 0;
}

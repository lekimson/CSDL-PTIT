#include <iostream>
#include <stack>
#include <string>
using namespace std;

// Chuyển đổi biểu thức tiền tố sang trung tố
string prefixToInfix(string exp) {
    stack<string> s;
    int n = exp.length();
    
    for (int i = n - 1; i >= 0; i--) {
        char ch = exp[i];
        
        if (isalnum(ch)) {
            s.push(string(1, ch));
        } else {
            string op1 = s.top(); s.pop();
            string op2 = s.top(); s.pop();
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
        cout << prefixToInfix(exp) << endl;
    }
    return 0;
}
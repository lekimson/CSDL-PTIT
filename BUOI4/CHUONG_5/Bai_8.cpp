#include <iostream>
#include <stack>
#include <string>
using namespace std;

// Chuyển đổi biểu thức tiền tố sang hậu tố
string prefixToPostfix(string exp) {
    stack<string> s;
    int n = exp.length();
    
    for (int i = n - 1; i >= 0; i--) {
        char ch = exp[i];
        
        if (isalnum(ch)) {
            s.push(string(1, ch));
        } else {
            string op1 = s.top(); s.pop();
            string op2 = s.top(); s.pop();
            string newExp = op1 + op2 + ch;
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
        cout << prefixToPostfix(exp) << endl;
    }
    return 0;
}
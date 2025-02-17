#include <iostream>
#include <stack>
using namespace std;

string simplifyExpression(string P) {
    stack<int> signStack;
    signStack.push(1); // Mặc định dấu ngoài cùng là dương
    int currentSign = 1;
    string result = "";
    
    for (size_t i = 0; i < P.size(); i++) {
        if (P[i] == '(') {
            if (i > 0 && P[i - 1] == '-') {
                signStack.push(signStack.top() * -1);
            } else {
                signStack.push(signStack.top());
            }
        } else if (P[i] == ')') {
            signStack.pop();
        } else if (P[i] == '+' || P[i] == '-') {
            if (i == 0 || P[i - 1] == '(') {
                result += (P[i] == '-') ? '-' : '+';
            } else {
                int sign = (P[i] == '-') ? -1 : 1;
                if (signStack.top() == -1) {
                    sign = -sign;
                }
                result += (sign == 1) ? '+' : '-';
            }
        } else {
            result += P[i];
        }
    }
    
    // Loại bỏ dấu '+' dư thừa ở đầu biểu thức
    if (!result.empty() && result[0] == '+') {
        result = result.substr(1);
    }
    
    return result;
}

int main() {
    int T;
    cin >> T;
    cin.ignore(); // Loại bỏ ký tự xuống dòng sau khi nhập số bộ test
    
    while (T--) {
        string P;
        getline(cin, P);
        cout << simplifyExpression(P) << endl;
    }
    
    return 0;
}
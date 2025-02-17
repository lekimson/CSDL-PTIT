#include <iostream>
#include <stack>
#include <string>
using namespace std;

string normalizeExpression(string P) {
    string result = "";
    stack<bool> s;
    bool flip = false;
    
    for (int i = 0; i < P.length(); i++) {
        if (P[i] == '(') {
            if (i > 0 && P[i - 1] == '-') {
                flip = !flip;
            }
            s.push(flip);
        } else if (P[i] == ')') {
            if (!s.empty()) {
                flip = s.top();
                s.pop();
            }
        } else if (P[i] == '+' || P[i] == '-') {
            if (flip) {
                result += (P[i] == '+' ? '-' : '+');
            } else {
                result += P[i];
            }
        } else {
            result += P[i];
        }
    }
    return result;
}

int main() {
    int T;
    cin >> T;
    cin.ignore();
    while (T--) {
        string P1, P2;
        getline(cin, P1);
        getline(cin, P2);
        
        if (normalizeExpression(P1) == normalizeExpression(P2)) {
            cout << "YES" << endl;
        } else {
            cout << "NO" << endl;
        }
    }
    return 0;
}

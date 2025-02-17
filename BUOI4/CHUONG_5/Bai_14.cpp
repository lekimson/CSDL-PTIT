#include <iostream>
#include <stack>
#include <string>
using namespace std;

string getSmallestNumber(string s) {
    string result = "";
    stack<int> st;
    int num = 1;
    
    for (int i = 0; i <= s.length(); i++) {
        st.push(num++);
        
        if (i == s.length() || s[i] == 'I') {
            while (!st.empty()) {
                result += to_string(st.top());
                st.pop();
            }
        }
    }
    
    return result;
}

int main() {
    int T;
    cin >> T;
    cin.ignore();
    while (T--) {
        string s;
        getline(cin, s);
        cout << getSmallestNumber(s) << endl;
    }
    return 0;
}
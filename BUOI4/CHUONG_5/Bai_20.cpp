#include <iostream>
#include <stack>
using namespace std;

string decodeString(const string &s) {
    stack<int> countStack;
    stack<string> stringStack;
    string currentString = "";
    int currentNum = 0;
    
    for (char c : s) {
        if (isdigit(c)) {
            currentNum = currentNum * 10 + (c - '0');
        } else if (c == '[') {
            countStack.push(currentNum);
            stringStack.push(currentString);
            currentNum = 0;
            currentString = "";
        } else if (c == ']') {
            string temp = currentString;
            int repeatTimes = countStack.top(); countStack.pop();
            currentString = stringStack.top(); stringStack.pop();
            
            while (repeatTimes--) {
                currentString += temp;
            }
        } else {
            currentString += c;
        }
    }
    
    return currentString;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        string str;
        cin >> str;
        cout << decodeString(str) << endl;
    }
    return 0;
}
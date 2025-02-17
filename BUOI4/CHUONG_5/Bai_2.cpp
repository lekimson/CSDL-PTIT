#include <iostream>
#include <string>
using namespace std;

int minReversalsToBalance(string s) {
    int left = 0, right = 0;
    for (char ch : s) {
        if (ch == '(') {
            left++;
        } else {
            if (left > 0) {
                left--;
            } else {
                right++;
            }
        }
    }
    return (left + 1) / 2 + (right + 1) / 2;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        string S;
        cin >> S;
        cout << minReversalsToBalance(S) << endl;
    }
    return 0;
}
#include <iostream>
#include <queue>
using namespace std;

void generateBinaryNumbers(int n) {
    queue<string> q;
    q.push("1");
    
    for (int i = 1; i <= n; i++) {
        string bin = q.front();
        q.pop();
        cout << bin << " ";
        q.push(bin + "0");
        q.push(bin + "1");
    }
    cout << endl;
}

int main() {
    int T, n;
    cin >> T;
    while (T--) {
        cin >> n;
        generateBinaryNumbers(n);
    }
    return 0;
}

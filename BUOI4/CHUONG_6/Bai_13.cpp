#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

bool compareLocPhat(const string &a, const string &b) {
    if (a.length() == b.length()) return a < b;  // Sắp xếp theo giá trị số
    return a.length() < b.length();  // Sắp xếp theo độ dài trước
}

vector<string> generateLocPhatNumbers(int N) {
    vector<string> result;
    queue<string> q;
    q.push("6");
    q.push("8");

    while (!q.empty()) {
        string num = q.front();
        q.pop();

        if (num.length() > N) break;  // Giới hạn số chữ số

        result.push_back(num);
        q.push(num + "6");
        q.push(num + "8");
    }

    sort(result.begin(), result.end(), compareLocPhat); // Sắp xếp theo độ dài và giá trị số
    return result;
}

int main() {
    int T;
    cin >> T;
    
    while (T--) {
        int N;
        cin >> N;

        vector<string> locPhatNumbers = generateLocPhatNumbers(N);
        cout << locPhatNumbers.size() << endl;
        for (const string &num : locPhatNumbers) {
            cout << num << " ";
        }
        cout << endl;
    }

    return 0;
}

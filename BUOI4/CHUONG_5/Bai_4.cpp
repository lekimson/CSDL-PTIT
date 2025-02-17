#include <iostream>
#include <stack>
#include <set>
#include <vector>
#include <algorithm>
using namespace std;

set<string> results;
vector<pair<int, int>> brackets;

// Hàm sinh tất cả các tập hợp con của dấu ngoặc
void generateExpressions(string &expr, vector<bool> &remove, int index) {
    if (index == brackets.size()) {
        string newExpr = "";
        for (int i = 0; i < expr.length(); i++) {
            bool toRemove = false;
            for (int j = 0; j < brackets.size(); j++) {
                if (remove[j] && (i == brackets[j].first || i == brackets[j].second)) {
                    toRemove = true;
                    break;
                }
            }
            if (!toRemove) {
                newExpr += expr[i];
            }
        }
        if (newExpr != expr) results.insert(newExpr);
        return;
    }
    
    // Không xóa cặp ngoặc này
    remove[index] = false;
    generateExpressions(expr, remove, index + 1);
    
    // Xóa cặp ngoặc này
    remove[index] = true;
    generateExpressions(expr, remove, index + 1);
}

int main() {
    string expr;
    cin >> expr;
    stack<int> s;
    
    // Xác định các cặp ngoặc tương ứng
    for (int i = 0; i < expr.length(); i++) {
        if (expr[i] == '(') {
            s.push(i);
        } else if (expr[i] == ')') {
            if (!s.empty()) {
                brackets.push_back({s.top(), i});
                s.pop();
            }
        }
    }
    
    vector<bool> remove(brackets.size(), false);
    generateExpressions(expr, remove, 0);
    
    // In kết quả theo thứ tự từ điển
    for (const string &res : results) {
        cout << res << endl;
    }
    
    return 0;
}
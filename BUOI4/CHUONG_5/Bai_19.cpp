#include <iostream>
#include <stack>
#include <vector>
using namespace std;

vector<int> calculateSpan(vector<int>& prices, int n) {
    stack<int> st;
    vector<int> span(n);
    
    for (int i = 0; i < n; i++) {
        while (!st.empty() && prices[st.top()] <= prices[i]) {
            st.pop();
        }
        
        span[i] = (st.empty()) ? (i + 1) : (i - st.top());
        st.push(i);
    }
    
    return span;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int N;
        cin >> N;
        vector<int> prices(N);
        for (int i = 0; i < N; i++) {
            cin >> prices[i];
        }
        
        vector<int> result = calculateSpan(prices, N);
        for (int r : result) {
            cout << r << " ";
        }
        cout << endl;
    }
    return 0;
}

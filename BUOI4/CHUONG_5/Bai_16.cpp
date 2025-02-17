#include <iostream>
#include <stack>
#include <vector>
using namespace std;

long long getMaxRectangleArea(vector<int>& heights) {
    stack<int> st;
    long long maxArea = 0;
    int n = heights.size();
    
    for (int i = 0; i <= n; i++) {
        while (!st.empty() && (i == n || heights[st.top()] > heights[i])) {
            int h = heights[st.top()];
            st.pop();
            int w = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, (long long)h * w);
        }
        st.push(i);
    }
    
    return maxArea;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int N;
        cin >> N;
        vector<int> heights(N);
        for (int i = 0; i < N; i++) {
            cin >> heights[i];
        }
        cout << getMaxRectangleArea(heights) << endl;
    }
    return 0;
}

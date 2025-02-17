#include <iostream>
#include <vector>
#include <unordered_map>
#include <stack>
using namespace std;

vector<int> findNextGreaterFrequency(vector<int>& arr) {
    int n = arr.size();
    unordered_map<int, int> freq;
    vector<int> result(n, -1);
    stack<int> st;
    
    for (int num : arr) {
        freq[num]++;
    }
    
    for (int i = n - 1; i >= 0; --i) {
        while (!st.empty() && freq[arr[st.top()]] <= freq[arr[i]]) {
            st.pop();
        }
        
        if (!st.empty()) {
            result[i] = arr[st.top()];
        }
        
        st.push(i);
    }
    
    return result;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int n;
        cin >> n;
        vector<int> arr(n);
        for (int i = 0; i < n; i++) {
            cin >> arr[i];
        }
        
        vector<int> result = findNextGreaterFrequency(arr);
        for (int num : result) {
            cout << num << " ";
        }
        cout << endl;
    }
    return 0;
}
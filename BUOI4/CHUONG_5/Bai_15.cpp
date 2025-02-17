#include <iostream>
#include <vector>
#include <stack>
using namespace std;

vector<int> findNextSmallerOfNextGreater(vector<int>& arr, int n) {
    vector<int> nextGreater(n, -1);
    vector<int> nextSmaller(n, -1);
    stack<int> st;
    
    // Tìm phần tử lớn hơn tiếp theo
    for (int i = 0; i < n; i++) {
        while (!st.empty() && arr[st.top()] < arr[i]) {
            nextGreater[st.top()] = i;
            st.pop();
        }
        st.push(i);
    }
    
    // Xóa stack để sử dụng lại
    while (!st.empty()) st.pop();
    
    // Tìm phần tử nhỏ hơn tiếp theo của phần tử lớn hơn tiếp theo
    for (int i = 0; i < n; i++) {
        if (nextGreater[i] != -1) {
            int idx = nextGreater[i];
            for (int j = idx + 1; j < n; j++) {
                if (arr[j] < arr[idx]) {
                    nextSmaller[i] = arr[j];
                    break;
                }
            }
        }
    }
    
    return nextSmaller;
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
        vector<int> result = findNextSmallerOfNextGreater(arr, n);
        for (int num : result) {
            cout << num << " ";
        }
        cout << endl;
    }
    return 0;
}

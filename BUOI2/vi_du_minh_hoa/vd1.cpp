#include <iostream>
using namespace std;

int sum_1_to_n(int n) {
    int s = 0; // Initialize sum to 0
    for(int i = 1; i <= n; i++) { // Loop from 1 to n
        s += i; // Add i to the sum
    }
    return s; // Return the total sum
}

int main() {
    int n = 5;
    cout << "Tong 1..n = " << sum_1_to_n(n) << endl; // Output: 15
    return 0;
}
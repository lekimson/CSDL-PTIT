#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>

using namespace std;

const int MIN_PRIME = 1000;
const int MAX_PRIME = 9999;
vector<bool> isPrime(MAX_PRIME + 1, true);

// Sàng Eratosthenes để tìm số nguyên tố 4 chữ số
void sieve() {
    isPrime[0] = isPrime[1] = false;
    for (int i = 2; i * i <= MAX_PRIME; ++i) {
        if (isPrime[i]) {
            for (int j = i * i; j <= MAX_PRIME; j += i) {
                isPrime[j] = false;
            }
        }
    }
}

// Sinh các số nguyên tố có thể đạt được bằng cách đổi một chữ số
vector<int> getAdjacentPrimes(int num) {
    vector<int> adjPrimes;
    string numStr = to_string(num);
    for (int i = 0; i < 4; ++i) {
        char original = numStr[i];
        for (char digit = '0'; digit <= '9'; ++digit) {
            if (digit == original) continue;
            numStr[i] = digit;
            int newNum = stoi(numStr);
            if (newNum >= MIN_PRIME && isPrime[newNum]) {
                adjPrimes.push_back(newNum);
            }
        }
        numStr[i] = original;
    }
    return adjPrimes;
}

// BFS để tìm số bước biến đổi nhỏ nhất
int minTransformSteps(int S, int T) {
    if (S == T) return 0;
    queue<pair<int, int>> q;
    unordered_set<int> visited;
    q.push({S, 0});
    visited.insert(S);

    while (!q.empty()) {
        auto [current, steps] = q.front();
        q.pop();
        
        for (int neighbor : getAdjacentPrimes(current)) {
            if (neighbor == T) return steps + 1;
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                q.push({neighbor, steps + 1});
            }
        }
    }
    return -1; // Không tìm thấy đường đi
}

int main() {
    sieve(); // Tiền xử lý các số nguyên tố
    int T;
    cin >> T;
    while (T--) {
        int S, T;
        cin >> S >> T;
        cout << minTransformSteps(S, T) << endl;
    }
    return 0;
}
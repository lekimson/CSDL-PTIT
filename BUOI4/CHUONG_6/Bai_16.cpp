#include <iostream>
#include <vector>
#include <queue>

using namespace std;

struct Position {
    int x, y, steps;
};

int minStepsToReachEnd(vector<vector<int>> &A, int M, int N) {
    vector<vector<bool>> visited(M, vector<bool>(N, false));
    queue<Position> q;
    q.push({0, 0, 0});
    visited[0][0] = true;
    
    while (!q.empty()) {
        Position current = q.front();
        q.pop();
        
        int i = current.x, j = current.y, steps = current.steps;
        if (i == M - 1 && j == N - 1) return steps;
        
        int move = A[i][j];
        
        if (j + move < N && !visited[i][j + move]) {
            visited[i][j + move] = true;
            q.push({i, j + move, steps + 1});
        }
        
        if (i + move < M && !visited[i + move][j]) {
            visited[i + move][j] = true;
            q.push({i + move, j, steps + 1});
        }
    }
    return -1;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int M, N;
        cin >> M >> N;
        vector<vector<int>> A(M, vector<int>(N));
        
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                cin >> A[i][j];
            }
        }
        
        cout << minStepsToReachEnd(A, M, N) << endl;
    }
    return 0;
}

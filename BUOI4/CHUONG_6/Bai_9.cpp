#include <iostream>
#include <vector>
#include <queue>
using namespace std;

struct Point {
    int x, y, steps;
};

int minMoves(vector<vector<char>> &grid, int a, int b, int c, int d) {
    int N = grid.size();
    vector<vector<int>> visited(N, vector<int>(N, 0));
    queue<Point> q;
    q.push({a, b, 0});
    visited[a][b] = 1;
    
    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};
    
    while (!q.empty()) {
        Point curr = q.front();
        q.pop();
        
        if (curr.x == c && curr.y == d) {
            return curr.steps;
        }
        
        for (int dir = 0; dir < 4; ++dir) {
            int nx = curr.x, ny = curr.y;
            while (nx + dx[dir] >= 0 && nx + dx[dir] < N && ny + dy[dir] >= 0 && ny + dy[dir] < N 
                   && grid[nx + dx[dir]][ny + dy[dir]] != 'X') {
                nx += dx[dir];
                ny += dy[dir];
                if (visited[nx][ny]) continue;
                visited[nx][ny] = 1;
                q.push({nx, ny, curr.steps + 1});
            }
        }
    }
    return -1;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int N;
        cin >> N;
        vector<vector<char>> grid(N, vector<char>(N));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cin >> grid[i][j];
            }
        }
        int a, b, c, d;
        cin >> a >> b >> c >> d;
        cout << minMoves(grid, a, b, c, d) << endl;
    }
    return 0;
}

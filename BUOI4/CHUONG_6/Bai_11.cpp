#include <iostream>
#include <vector>
#include <queue>
using namespace std;

struct Cell {
    int x, y, z, steps;
};

int minStepsToExit(vector<vector<vector<char>>> &grid, int A, int B, int C, Cell start, Cell end) {
    queue<Cell> q;
    q.push(start);
    vector<vector<vector<bool>>> visited(A, vector<vector<bool>>(B, vector<bool>(C, false)));
    visited[start.x][start.y][start.z] = true;
    
    int dx[] = {-1, 1, 0, 0, 0, 0};
    int dy[] = {0, 0, -1, 1, 0, 0};
    int dz[] = {0, 0, 0, 0, -1, 1};
    
    while (!q.empty()) {
        Cell curr = q.front();
        q.pop();
        
        if (curr.x == end.x && curr.y == end.y && curr.z == end.z) {
            return curr.steps;
        }
        
        for (int dir = 0; dir < 6; ++dir) {
            int nx = curr.x + dx[dir];
            int ny = curr.y + dy[dir];
            int nz = curr.z + dz[dir];
            
            if (nx >= 0 && nx < A && ny >= 0 && ny < B && nz >= 0 && nz < C 
                && grid[nx][ny][nz] != '#' && !visited[nx][ny][nz]) {
                visited[nx][ny][nz] = true;
                q.push({nx, ny, nz, curr.steps + 1});
            }
        }
    }
    
    return -1;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int A, B, C;
        cin >> A >> B >> C;
        vector<vector<vector<char>>> grid(A, vector<vector<char>>(B, vector<char>(C)));
        Cell start, end;
        
        for (int i = 0; i < A; ++i) {
            for (int j = 0; j < B; ++j) {
                for (int k = 0; k < C; ++k) {
                    cin >> grid[i][j][k];
                    if (grid[i][j][k] == 'S') start = {i, j, k, 0};
                    if (grid[i][j][k] == 'E') end = {i, j, k, 0};
                }
            }
        }
        
        cout << minStepsToExit(grid, A, B, C, start, end) << endl;
    }
    return 0;
}

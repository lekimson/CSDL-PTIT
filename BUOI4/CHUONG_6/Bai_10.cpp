#include <iostream>
#include <vector>
#include <queue>
using namespace std;

struct Cell {
    int x, y, days;
};

int minDaysToGerminate(vector<vector<int>> &grid, int R, int C) {
    queue<Cell> q;
    int totalSeeds = 0, germinatedSeeds = 0, days = 0;
    
    // Đếm số lượng hạt mầm và đưa cây non vào queue
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            if (grid[i][j] == 1 || grid[i][j] == 2) {
                totalSeeds++;
            }
            if (grid[i][j] == 2) {
                q.push({i, j, 0});
                germinatedSeeds++;
            }
        }
    }
    
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    
    // BFS để lan truyền chất dinh dưỡng
    while (!q.empty()) {
        Cell curr = q.front();
        q.pop();
        
        days = curr.days;
        
        for (int dir = 0; dir < 4; ++dir) {
            int nx = curr.x + dx[dir];
            int ny = curr.y + dy[dir];
            
            if (nx >= 0 && nx < R && ny >= 0 && ny < C && grid[nx][ny] == 1) {
                grid[nx][ny] = 2;
                q.push({nx, ny, curr.days + 1});
                germinatedSeeds++;
            }
        }
    }
    
    return (germinatedSeeds == totalSeeds) ? days : -1;
}

int main() {
    int T;
    cin >> T;
    while (T--) {
        int R, C;
        cin >> R >> C;
        vector<vector<int>> grid(R, vector<int>(C));
        
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) {
                cin >> grid[i][j];
            }
        }
        
        cout << minDaysToGerminate(grid, R, C) << endl;
    }
    return 0;
}
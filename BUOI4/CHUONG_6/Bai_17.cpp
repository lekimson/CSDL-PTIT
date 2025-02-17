#include <iostream>
#include <queue>
#include <unordered_map>

using namespace std;

struct Position {
    int x, y, steps;
};

const int boardSize = 8;
const int dx[8] = {-2, -1, 1, 2, 2, 1, -1, -2};
const int dy[8] = {1, 2, 2, 1, -1, -2, -2, -1};

// Chuyển đổi vị trí dạng "a1" thành tọa độ số (x, y)
pair<int, int> convertPosition(string pos) {
    return {pos[0] - 'a', pos[1] - '1'};
}

// Tìm số bước ít nhất để di chuyển quân mã từ start đến end
int minKnightMoves(string start, string end) {
    if (start == end) return 0;
    
    auto [sx, sy] = convertPosition(start);
    auto [ex, ey] = convertPosition(end);
    
    queue<Position> q;
    bool visited[boardSize][boardSize] = {};
    
    q.push({sx, sy, 0});
    visited[sx][sy] = true;
    
    while (!q.empty()) {
        Position curr = q.front();
        q.pop();
        
        for (int i = 0; i < 8; ++i) {
            int nx = curr.x + dx[i];
            int ny = curr.y + dy[i];
            
            if (nx == ex && ny == ey) return curr.steps + 1;
            
            if (nx >= 0 && nx < boardSize && ny >= 0 && ny < boardSize && !visited[nx][ny]) {
                visited[nx][ny] = true;
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
        string start, end;
        cin >> start >> end;
        cout << minKnightMoves(start, end) << endl;
    }
    return 0;
}
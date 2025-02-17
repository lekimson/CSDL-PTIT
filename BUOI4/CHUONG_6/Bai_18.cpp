#include <iostream>
#include <queue>
#include <map>
#include <vector>

using namespace std;

struct State {
    vector<int> board;
    int moves;
};

vector<int> rotateLeft(const vector<int>& board) {
    return {board[3], board[0], board[2], board[4], board[1], board[5]};
}

vector<int> rotateRight(const vector<int>& board) {
    return {board[0], board[4], board[1], board[3], board[5], board[2]};
}

int minMovesToTarget(vector<int> start, vector<int> target) {
    if (start == target) return 0;
    
    queue<State> q;
    map<vector<int>, int> visited;
    
    q.push({start, 0});
    visited[start] = 0;
    
    while (!q.empty()) {
        State current = q.front(); q.pop();
        
        vector<int> leftRotated = rotateLeft(current.board);
        if (leftRotated == target) return current.moves + 1;
        if (!visited.count(leftRotated)) {
            visited[leftRotated] = current.moves + 1;
            q.push({leftRotated, current.moves + 1});
        }
        
        vector<int> rightRotated = rotateRight(current.board);
        if (rightRotated == target) return current.moves + 1;
        if (!visited.count(rightRotated)) {
            visited[rightRotated] = current.moves + 1;
            q.push({rightRotated, current.moves + 1});
        }
    }
    
    return -1;
}

int main() {
    int t;
    cin >> t;
    while (t--) {
        vector<int> start(6), target(6);
        for (int i = 0; i < 6; i++) cin >> start[i];
        for (int i = 0; i < 6; i++) cin >> target[i];
        cout << minMovesToTarget(start, target) << endl;
    }
    return 0;
}
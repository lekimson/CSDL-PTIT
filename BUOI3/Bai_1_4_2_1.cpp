#include <iostream>
#include <stack>
#include <string>
using namespace std;

int main() {
    stack<string> stack;

    // Push các phần tử vào ngăn xếp
    stack.push("Sách A");
    cout << "Đã thêm 'Sách A' vào ngăn xếp." << endl;
    stack.push("Sách B");
    cout << "Đã thêm 'Sách B' vào ngăn xếp." << endl;
    stack.push("Sách C");
    cout << "Đã thêm 'Sách C' vào ngăn xếp." << endl;

    // Hiển thị phần tử ở đỉnh ngăn xếp
    cout << "Phần tử ở đỉnh ngăn xếp: " << stack.top() << endl; // Output: Sách C

    // Pop phần tử khỏi ngăn xếp
    stack.pop();
    cout << "Đã lấy phần tử ra khỏi ngăn xếp." << endl;

    // Hiển thị phần tử ở đỉnh ngăn xếp sau khi pop
    cout << "Phần tử ở đỉnh ngăn xếp sau khi pop: " << stack.top() << endl; // Output: Sách B

    // Kiểm tra ngăn xếp có trống không
    if (stack.empty()) {
        cout << "Ngăn xếp trống!" << endl;
    } else {
        cout << "Ngăn xếp không trống." << endl; // Output: Ngăn xếp không trống.
    }

    return 0;
}

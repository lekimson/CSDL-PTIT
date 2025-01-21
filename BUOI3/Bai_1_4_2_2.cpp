// custom_stack_cpp.cpp
#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Stack {
private:
    vector<string> elements; // Sử dụng vector để lưu trữ phần tử

public:
    // Thêm phần tử vào ngăn xếp
    void push(const string& item) {
        elements.push_back(item);
        cout << "Đã thêm '" << item << "' vào ngăn xếp." << endl;
    }

    // Loại bỏ phần tử khỏi ngăn xếp
    void pop() {
        if (!isEmpty()) {
            string item = elements.back();
            elements.pop_back();
            cout << "Đã lấy '" << item << "' ra khỏi ngăn xếp." << endl;
        } else {
            cout << "Ngăn xếp trống!" << endl;
        }
    }

    // Xem phần tử ở đỉnh ngăn xếp
    string peek() const {
        if (!isEmpty()) {
            return elements.back();
        } else {
            throw out_of_range("Ngăn xếp trống!");
        }
    }

    // Kiểm tra ngăn xếp rỗng
    bool isEmpty() const {
        return elements.empty();
    }

    // Trả về kích thước ngăn xếp
    int size() const {
        return elements.size();
    }

    // In nội dung ngăn xếp từ đỉnh đến đáy
    void display() const {
        cout << "Ngăn xếp (đỉnh đến đáy): ";
        for (auto it = elements.rbegin(); it != elements.rend(); ++it) {
            cout << *it << " ";
        }
        cout << endl;
    }
};

// Hàm main để minh họa
int main() {
    Stack stack;
    stack.push("Sách A");
    stack.push("Sách B");
    stack.push("Sách C");
    stack.display(); // Output: Ngăn xếp (đỉnh đến đáy): Sách C Sách B Sách A

    try {
        cout << "Phần tử ở đỉnh ngăn xếp: " << stack.peek() << endl; // Output: Sách C
    } catch (const out_of_range& e) {
        cout << e.what() << endl;
    }

    stack.pop();
    stack.display(); // Output: Ngăn xếp (đỉnh đến đáy): Sách B Sách A

    cout << "Ngăn xếp có trống không? " << (stack.isEmpty() ? "Có" : "Không") << endl; // Output: Không

    return 0;
}

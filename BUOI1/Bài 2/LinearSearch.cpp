#include <iostream>
#include <vector>
using namespace std; 

// A: vector chứa các phần tử cần tìm
// x: giá trị cần tìm kiếm
int LinearSearch(const vector<int>& A, int x) {
    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i] == x) {
            return static_cast<int>(i); // Trả về chỉ số của phần tử
        }
    }
    return -1; // Không tìm thấy
}

int main() {
    // Khởi tạo vector A và giá trị cần tìm x
    vector<int> A = {3, 5, 2, 9, 1};
    int x = 9;

    // Gọi hàm LinearSearch và lưu kết quả vào biến index
    int index = LinearSearch(A, x);

    // Kiểm tra và in ra kết quả
    if (index != -1) {
        cout << "The value " << x << " just found at the location: " << index << endl;
    } else {
        cout << "The value " << x << " does not exist in the array." << endl;
    }

    return 0; // Kết thúc chương trình
}

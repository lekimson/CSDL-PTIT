#include <iostream> 
#include <vector>    
#include <limits.h>  
using namespace std;

// Hàm FindMax: Tìm giá trị lớn nhất trong một mảng (vector) các số nguyên
int FindMax(const vector<int>& A) {
    // Kiểm tra nếu mảng rỗng
    if (A.empty()) {
        cerr << "Error: Empty array!" << endl; // Thông báo lỗi
        return INT_MIN; // Trả về giá trị nhỏ nhất của kiểu int để báo lỗi
    }

    //  phần tử đầu tiên là giá trị lớn nhất
    int max = A[0];

    // Lặp qua các phần tử còn lại của mảng để tìm giá trị lớn nhất
    for (size_t i = 1; i < A.size(); ++i) {
        if (A[i] > max) { //  phần tử hiện tại lớn hơn giá trị lớn nhất hiện tại
            max = A[i];   // Cập nhật giá trị lớn nhất
        }
    }

    return max; // Trả về giá trị lớn nhất
}

int main() {
    // Khởi tạo một mảng (vector) chứa các số nguyên
    vector<int> A = {3, 5, 2, 9, 1};

    // Gọi hàm FindMax để tìm giá trị lớn nhất trong mảng
    int maxVal = FindMax(A);

    // Nếu giá trị trả về không phải là INT_MIN, in ra kết quả
    if (maxVal != INT_MIN) {
        cout << "The largest number in the sequence is " << maxVal << endl;
    }

    return 0;
}

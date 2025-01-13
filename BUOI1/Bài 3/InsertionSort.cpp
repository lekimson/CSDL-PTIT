#include <iostream>
#include <vector>
 using namespace std;
// Hàm thực hiện thuật toán sắp xếp chèn (Insertion Sort)
void InsertionSort(std::vector<int>& A) {
    // Lặp qua từng phần tử của mảng bắt đầu từ vị trí thứ 1
    for (size_t i = 1; i < A.size(); ++i) {
        int key = A[i];                // Lưu giá trị hiện tại
        int j = static_cast<int>(i) - 1; // Chỉ số phần tử trước đó

        // Dịch chuyển các phần tử lớn hơn giá trị hiện tại  sang phải
        while (j >= 0 && A[j] > key) {
            A[j + 1] = A[j];
            --j;
        }

        // Chèn giá trị vào vị trí đúng
        A[j + 1] = key;
    }
}

int main() {
   

    vector<int> A = {5, 2, 9, 1, 5, 6}; // Khởi tạo mảng ban đầu

    InsertionSort(A); // Gọi hàm sắp xếp chèn

    // Xuất mảng sau khi sắp xếp
    cout << "The array after sorting: ";
    for (const auto& num : A) {
        cout << num << " "; // Xuất từng phần tử
    }
    cout << endl;

    return 0;
}

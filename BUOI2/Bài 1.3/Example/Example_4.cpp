//đây là chương trình sử dụng merge sort để sắp xếp mảng
#include <iostream>
#include <vector>

using namespace std;

// Hàm trộn hai mảng đã sắp xếp
void merge(vector<int> &arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Tạo mảng tạm
    vector<int> L(n1), R(n2);

    // Sao chép dữ liệu vào mảng tạm L và R
    for (int i = 0; i < n1; i++) {
        L[i] = arr[left + i];
    }
    for (int j = 0; j < n2; j++) {
        R[j] = arr[mid + 1 + j];
    }

    // Gộp hai mảng tạm L và R vào mảng arr
    int i = 0; // Chỉ số của mảng con đầu tiên
    int j = 0; // Chỉ số của mảng con thứ hai
    int k = left; // Chỉ số của mảng đã sắp xếp
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }

    // Sao chép các phần tử còn lại của mảng L nếu có
    while (i < n1) {
        arr[k++] = L[i++];
    }

    // Sao chép các phần tử còn lại của mảng R nếu có
    while (j < n2) {
        arr[k++] = R[j++];
    }
}

// Hàm sắp xếp trộn
void mergeSort(vector<int> &arr, int left, int right) {
    if (left < right) {
        // Tìm điểm giữa
        int mid = left + (right - left) / 2;

        // Sắp xếp nửa đầu và nửa sau
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // Trộn hai nửa
        merge(arr, left, mid, right);
    }
}

int main() {
    vector<int> arr = {5, 2, 9, 1, 5, 6};
    int n = arr.size();

    // Gọi hàm sắp xếp
    mergeSort(arr, 0, n - 1);

    // In mảng đã sắp xếp
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    return 0;
}

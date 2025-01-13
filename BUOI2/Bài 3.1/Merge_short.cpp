// đây là chương trình sắp xếp mảng dựa trên thuật toán merge short
#include <iostream>
#include <vector>

using namespace std;

// Hàm Merge_Short thực hiện việc trộn 2 mảng con
void Merge_Short(vector<int>& arr, int left, int right, int mid)
{
    // Tính độ dài của 2 mảng con
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Tạo mảng tạm thời để lưu trữ các phần tử từ mảng con
    vector<int> L(n1), R(n2);

    // Copy dữ liệu từ mảng arr vào mảng tạm thời L và R
    for (int i = 0; i < n1; i++) {
        L[i] = arr[left + i];
    }
    for (int j = 0; j < n2; j++) {
        R[j] = arr[mid + 1 + j];
    }

    // Trộn các phần tử của 2 mảng con vào mảng arr
    int i = 0, j = 0;
    int k = left; // Chỉ số của mảng arr bắt đầu từ vị trí left
    while (i < n1 && j < n2) {
        // Nếu phần tử của mảng L nhỏ hơn hoặc bằng phần tử của mảng R
        if (L[i] <= R[j]) {
            arr[k++] = L[i++]; // Đưa phần tử L[i] vào arr
        } else {
            arr[k++] = R[j++]; // Đưa phần tử R[j] vào arr
        }
    }

    // Sao chép phần tử còn lại của mảng L vào arr (nếu có)
    while (i < n1) {
        arr[k++] = L[i++];
    }

    // Sao chép phần tử còn lại của mảng R vào arr (nếu có)
    while (j < n2) {
        arr[k++] = R[j++];
    }
}

// Hàm đệ quy thực hiện Merge Sort
void merge(vector<int>& arr, int left, int right)
{
    if (left < right)
    {
        // Tính giá trị mid
        int mid = left + (right - left) / 2;
        
        // Đệ quy chia mảng ra làm 2 nửa
        merge(arr, left, mid);
        merge(arr, mid + 1, right);

        // Gọi hàm Merge_Short để trộn mảng con
        Merge_Short(arr, left, right, mid);
    }
}

int main()
{
    // Khởi tạo mảng cần sắp xếp
    vector<int> arr = {5, 2, 9, 1, 5, 6};
    
    // Gọi hàm merge để sắp xếp mảng bằng Merge Sort
    merge(arr, 0, arr.size() - 1);
    
    // In kết quả sau khi sắp xếp
    cout << "Mang sau khi sap xep ";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;
    
    return 0;
}

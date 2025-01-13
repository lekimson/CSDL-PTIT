//đây là chương trình sắp xếp mảng dựa trên thuật toán quick short
#include <iostream>
#include <vector>

using namespace std;

// Hàm Partition
int partitionFunc(vector<int> &arr, int low, int high)
{
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) // Duyệt qua mảng từ `low` đến `high - 1`
    {
        if (arr[j] < pivot)
        {
            i++;
            // Hoán đổi thủ công không dùng std::swap
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    // Hoán đổi pivot với phần tử đứng sau `i`
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return i + 1; // Trả về vị trí của pivot
}

// Hàm Quick Sort
void quickSort(vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        // Tìm pivot sao cho mảng được chia thành 2 mảng con
        int pi = partitionFunc(arr, low, high);

        // Đệ quy sắp xếp 2 mảng con
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main()
{
    vector<int> arr = {5, 2, 9, 1, 6, 5};
    int n = arr.size();
    quickSort(arr, 0, n - 1);

    cout << "Sorted array: \n";
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << " ";
    }
    cout << endl;

    return 0;
}

#include <vector>
#include <algorithm> // std::swap
using namespace std;

// Hàm median_of_three
int median_of_three(vector<int> &arr, int low, int high)
{
    int mid = low + (high - low) / 2;

    // Sắp xếp ba giá trị tại low, mid, high
    if (arr[low] > arr[mid])
        swap(arr[low], arr[mid]);
    if (arr[low] > arr[high])
        swap(arr[low], arr[high]);
    if (arr[mid] > arr[high])
        swap(arr[mid], arr[high]);

    // Hoán đổi giá trị trung vị (median) vào vị trí high
    swap(arr[mid], arr[high]);

    // Trả về chỉ số high (pivot)
    return high;
}

// Hàm partition sử dụng median
int partition_median(vector<int> &arr, int low, int high)
{
    int median = median_of_three(arr, low, high);

    // Đặt giá trị trung vị vào vị trí pivot cuối
    swap(arr[median], arr[high]);

    // Tiến hành phân vùng
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++)
    {
        if (arr[j] <= pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Hàm Quick Sort sử dụng median
void quickShort_median(vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int pi = partition_median(arr, low, high);

        quickShort_median(arr, low, pi - 1);
        quickShort_median(arr, pi + 1, high);
    }
}

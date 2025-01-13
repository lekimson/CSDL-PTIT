#include <iostream>
#include <vector>
using namespace std;

// Biến toàn cục để đếm số lần so sánh
long long comparisonCount = 0;

// Merge Sort
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        comparisonCount++; // Đếm số lần so sánh
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Partition function for Quick Sort
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        comparisonCount++; // Đếm số lần so sánh
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Quick Sort
void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int p = partition(arr, low, high);
        quickSort(arr, low, p - 1);
        quickSort(arr, p + 1, high);
    }
}

// Hàm main
int main() {
    vector<int> arr = {12, 4, 5, 6, 7, 3, 1, 15, 2, 8, 10, 9};

    cout << "Mảng ban đầu: ";
    for (int x : arr) cout << x << " ";
    cout << endl;

    // Merge Sort
    comparisonCount = 0;
    vector<int> arr1 = arr;
    mergeSort(arr1, 0, arr1.size() - 1);
    cout << "Mảng sau khi sắp xếp (Merge Sort): ";
    for (int x : arr1) cout << x << " ";
    cout << endl;
    cout << "Số lần so sánh (Merge Sort): " << comparisonCount << endl;

    // Quick Sort
    comparisonCount = 0;
    vector<int> arr2 = arr;
    quickSort(arr2, 0, arr2.size() - 1);
    cout << "Mảng sau khi sắp xếp (Quick Sort): ";
    for (int x : arr2) cout << x << " ";
    cout << endl;
    cout << "Số lần so sánh (Quick Sort): " << comparisonCount << endl;

    return 0;
}

// chương trình sử dụng thuật toán bubble sort để sắp xếp mảng
#include <iostream>
#include <vector>

using namespace std;

// Hàm thực hiện sắp xếp mảng theo Bubble Sort
void bubble_sort(vector<int> &arr)
{
    int n = arr.size();
    bool swapped;
    for (int i = 0; i < n - 1; i++)
    {
        swapped = false;
        for (int j = 0; j < n - 1 - i; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                // Thực hiện hoán đổi
              int temp = arr[j];
              arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = true;
            }
        }
   
    }
}

int main()
{
    vector<int> arr = {5, 2, 9, 1, 6};
    bubble_sort(arr);

    cout << "Ket qua sau khi sap xep: ";
    for (int x : arr)
    {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}

//đây là chương trình C++ dùng thuật toán selection short

#include <iostream>
#include <vector>

using namespace std;

void Selection_Short (vector <int> &arr)
{
    int n = arr.size();
    for (int i = 0 ; i < n - 1; i++)
    {
        int min_index = i;
        for (int j = i + 1; j < n; j++)
        {
            if (arr[j] < arr[min_index])
            {
                min_index = j;
            }
        }
        if (min_index != i)
        {
            swap(arr[i], arr[min_index]);
        }
    }
}
int main()
{
    vector<int> arr = {5, 2, 9, 1, 6};
    Selection_Short(arr);

    cout << "Ket qua sau khi sap xep: ";
    for (int x : arr)
    {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
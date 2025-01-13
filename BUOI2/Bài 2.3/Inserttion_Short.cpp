//đây là chương trình sắp xếp sử dụng thuật toán insertion short

#include <iostream>
#include <vector>

using namespace std;

//hàm insertion short 
 void Insert_sort(vector<int> &arr)
{
    int n = arr.size();
    for (int i = 1; i < n; i++)
    {
        int key = arr[i];
        int j = i - 1;
        //di chuyển các phần tử lớn hơn sang bên phải 
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main ()
{
    vector<int> arr = {5, 2, 9, 1, 6};
    Insert_sort(arr);

    cout << "Ket qua sau khi sap xep: ";
    for (int x : arr)
    {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
//đây là chương trình c++ về vòng lặp đơn
#include <iostream>

using namespace std;

int sum_1_to_n(int n) {
    int sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += i;
    }
    return sum;
} 
int main ()

{

    int n = 5;
    cout << "The total sum of 1 to " << n << " is " << sum_1_to_n(n) << endl; //output = 15 (sai làm chó)
}
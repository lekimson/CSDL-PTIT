#include <iostream>
using namespace std;

int Inputnumber()
{
    int n;
    cout << "Enter a number: ";
    cin >> n;
    return n;
}
//moduel to calculate factorial
long long ComputeFactorial(int n)
{
    if ( n == 0)
    {
        return 1;
    }
    else
    {
        return n * ComputeFactorial(n-1);
    }

}
 //module to display factorial
 void OutputResult (long long result)
 {
     cout << "The factorial is: " << result << endl;
 }

 //program coordination
    void Factorial() {
    int n = Inputnumber();
    if (n < 0) {
        cout << "Error: n phải là số nguyên không âm." << std::endl;
        return;
    }
    long long result = ComputeFactorial(n);
    OutputResult(result);
}
int main()
{
    Factorial();
    return 0;
}
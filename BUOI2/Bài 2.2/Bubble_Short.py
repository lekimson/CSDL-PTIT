# đây là chương trình dùng thuật toán bubble short để thực hiện sắp xếp mảng arr

def bubble_short(arr):
    n = len(arr)
    for i in range(n- 1):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
if __name__ == "__main__":
    arr = [5, 2, 9, 1, 5, 6]
    bubble_short(arr)
    print("Kết quả Bubble Short:", arr)  # Output: [1, 2, 5, 5, 6, 9]   
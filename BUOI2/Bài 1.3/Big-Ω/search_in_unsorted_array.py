
#đây là chương trinh python tìm kiếm trong mảng không sắp xếp 
def find_first_element(arr, target):
        
            if arr[0] == target:
                return 0 #trả về giá trị bằng 0
            return -1
arr = [10, 20, 30, 40, 50]
print(find_first_element(arr, 10)) # 0
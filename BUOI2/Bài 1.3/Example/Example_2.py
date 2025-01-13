# đây là chương trình python về tìm kiếm nhị phân
def Linear_Search(arr, target):
    # Duyệt qua từng phần tử trong danh sách arr với chỉ số (index) và giá trị (value)
    for index, value in enumerate(arr):  
        # Nếu tìm thấy giá trị cần tìm, trả về chỉ số của nó
        if value == target:
            return index
    # Nếu không tìm thấy giá trị trong danh sách, trả về -1
    return -1

arr = [4, 2, 5, 1, 3]
# Tìm kiếm giá trị 5 trong mảng và in ra kết quả
print(Linear_Search(arr, 5))  # 2

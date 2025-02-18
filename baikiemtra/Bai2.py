#đây là chương trình đếm số lần xuất hiện của các ký tự trong chuỗi.
def count_chars(string):
    
    string = string.strip()  # Loại bỏ khoảng trắng ở đầu và cuối chuỗi.
    char_count = {}  # Khởi tạo từ điển để lưu kết quả.

    # Lặp qua từng ký tự trong chuỗi
    for char in string:
        if char.isalpha():  # Kiểm tra nếu ký tự là chữ cái (loại bỏ số & ký tự đặc biệt).
            # Nếu ký tự đã có trong từ điển thì tăng số lần xuất hiện, ngược lại gán giá trị ban đầu là 1.
            char_count[char] = char_count.get(char, 0) + 1  

    return char_count  # Trả về từ điển chứa số lần xuất hiện của từng ký tự.

# Test cases
string1 = "Happiness"
string2 = "smiles"

# In kết quả ra màn hình
print(count_chars(string1))  # {'H': 1, 'a': 1, 'p': 2, 'i': 1, 'n': 1, 'e': 1, 's': 2}
print(count_chars(string2))  # {'s': 2, 'm': 1, 'i': 1, 'l': 1, 'e': 1}

def word_count(file_path):
    word_freq = {}  # Dictionary lưu số lần xuất hiện của từng từ
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:  # Đọc từng dòng trong file
            words = line.strip().lower().split()  # Chuyển thành chữ thường và tách từ
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1  # Cập nhật số lần xuất hiện
    
    return word_freq
file_path = "./Session_5/P1_data.txt"
result = word_count(file_path)
print(result)
def quick_sort_records(records):
    if len(records) <= 1:
        return records
    else:
        pivot = records[-1]  # Chọn pivot là ngày đăng nhập cuối cùng
        left = [record for record in records[:-1] if record[1] <= pivot[1]]
        right = [record for record in records[:-1] if record[1] > pivot[1]]
        
        return quick_sort_records(right) + [pivot] + quick_sort_records(left)

# Danh sách bản ghi gồm tên người dùng và ngày đăng nhập (định dạng YYYY-MM-DD)
records = [
    ("user1", "2025-01-12"),
    ("user2", "2025-01-10"),
    ("user3", "2025-01-11"),
    ("user4", "2025-01-13"),
    ("user5", "2025-01-09")
]

# Sắp xếp danh sách bản ghi theo ngày đăng nhập mới nhất
sorted_records = quick_sort_records(records)

# In danh sách đã sắp xếp
for record in sorted_records:
    print(record)

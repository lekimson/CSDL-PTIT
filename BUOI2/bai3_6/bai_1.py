def merge_sort_students(students):
    if len(students) > 1:
        mid = len(students) // 2
        left_half = students[:mid]
        right_half = students[mid:]

        merge_sort_students(left_half)
        merge_sort_students(right_half)

        i = j = k = 0

        # So sánh và trộn các phần đã sắp xếp
        while i < len(left_half) and j < len(right_half):
            if left_half[i][1] < right_half[j][1]:  # So sánh điểm số
                students[k] = left_half[i]
                i += 1
            elif left_half[i][1] > right_half[j][1]:
                students[k] = right_half[j]
                j += 1
            else:  # Nếu điểm số bằng nhau, so sánh theo tên
                if left_half[i][0] <= right_half[j][0]:
                    students[k] = left_half[i]
                    i += 1
                else:
                    students[k] = right_half[j]
                    j += 1
            k += 1

        # Gộp các phần còn lại của mảng
        while i < len(left_half):
            students[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            students[k] = right_half[j]
            j += 1
            k += 1

# Danh sách sinh viên gồm tên và điểm số
students = [
    ("Nguyen Van A", 85),
    ("Le Thi B", 90),
    ("Tran Van C", 85),
    ("Pham Thi D", 92),
    ("Bui Van E", 85)
]

# Sắp xếp danh sách sinh viên
merge_sort_students(students)

# In danh sách đã sắp xếp
for student in students:
    print(student)

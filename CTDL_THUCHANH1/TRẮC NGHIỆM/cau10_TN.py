class MyStack:
    def __init__(self, capacity):
        self._capacity = capacity  # Khởi tạo capacity cho stack
        self.__stack = []  # Danh sách chứa các phần tử của stack
    
    def is_full(self):
        # Kiểm tra xem stack đã đầy chưa
        return len(self.__stack) == self._capacity
    
    def is_empty(self):
        # Kiểm tra xem stack có rỗng không
        return len(self.__stack) == 0
    
    def push(self, value):
        # Thêm phần tử vào stack nếu chưa đầy
        if not self.is_full():
            self.__stack.append(value)
        else:
            print("Stack is full!")
    
    def top(self):
        # Lấy phần tử trên cùng của stack mà không xóa nó
        if not self.is_empty():
            return self.__stack[-1]
        else:
            print("Stack is empty!")
            return None

# Tạo đối tượng stack với capacity là 5
stack1 = MyStack(capacity=5)

# Thêm phần tử vào stack
stack1.push(1)
stack1.push(2)

# Lấy phần tử trên cùng mà không xóa
print(stack1.top())  # In ra kết quả
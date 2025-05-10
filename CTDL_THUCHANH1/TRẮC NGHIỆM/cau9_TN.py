class MyStack:
    def __init__(self, capacity):
        self._capacity = capacity  # Khởi tạo capacity cho stack
        self.__stack = []  # Danh sách chứa các phần tử của stack
    
    def is_full(self):
        # Kiểm tra xem stack đã đầy chưa
        return len(self.__stack) == self._capacity
    
    def push(self, value):
        # Thêm phần tử vào stack nếu chưa đầy
        if not self.is_full():
            self.__stack.append(value)
        else:
            print("Stack is full!")

# Tạo đối tượng stack với capacity là 5
stack1 = MyStack(capacity=5)

# Thêm phần tử vào stack
stack1.push(1)
stack1.push(2)

# Kiểm tra xem stack đã đầy chưa
print(stack1.is_full())  # In ra kết quả
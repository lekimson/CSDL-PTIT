class Stack:
    def __init__(self, capacity):
        self.capacity = capacity  # Khoi tao dung luong stack
        self.stack = []  # Dung list de mo phong stack
    
    def is_empty(self):
        """Kiem tra stack co rong hay khong"""
        return len(self.stack) == 0
    
    def is_full(self):
        """Kiem tra stack co day khong"""
        return len(self.stack) == self.capacity
    
    def pop(self):
        """Loai bo va tra ve phan tu top cua stack"""
        if not self.is_empty():
            return self.stack.pop()  # Loai bo va tra ve phan tu cuoi cung
        else:
            print("Stack rong")
            return None
    
    def push(self, value):
        """Them phan tu vao stack neu stack chua day"""
        if not self.is_full():
            self.stack.append(value)  # Them phan tu vao cuoi list (top cua stack)
        else:
            print("Stack day")
    
    def top(self):
        """Lay gia tri top element ma khong loai bo no"""
        if not self.is_empty():
            return self.stack[-1]  # Tra ve phan tu cuoi cung cua list
        else:
            print("Stack rong")
            return None

# Vi du su dung
stack1 = Stack(capacity=5)

print("Them phan tu vao stack:")
stack1.push(1)
print(f"Top cua stack sau khi push(1): {stack1.top()}")

stack1.push(2)
print(f"Top cua stack sau khi push(2): {stack1.top()}")

# Kiem tra xem stack co day khong
print(f"\nKiem tra stack da day chua: {stack1.is_full()}")  # False

# Lay top element ma khong loai bo
print(f"Top cua stack: {stack1.top()}")  # 2

# Pop phan tu tu stack
print(f"\nLoai bo top element: {stack1.pop()}")  # 2
print(f"Top cua stack sau khi pop: {stack1.top()}")  # 1

# Pop tiep phan tu nua
print(f"\nLoai bo top element: {stack1.pop()}")  # 1

# Kiem tra xem stack co rong khong
print(f"\nKiem tra stack da rong chua: {stack1.is_empty()}")  # True
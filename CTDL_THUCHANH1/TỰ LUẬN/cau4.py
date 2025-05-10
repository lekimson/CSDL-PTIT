class MyQueue:
    def __init__(self, capacity):
        """Khởi tạo queue với capacity cho trước"""
        self.capacity = capacity  # Dung lượng tối đa của queue
        self.queue = []  # Dùng list để mô phỏng queue
    
    def is_empty(self):
        """Kiểm tra queue có rỗng hay không"""
        return len(self.queue) == 0
    
    def is_full(self):
        """Kiểm tra queue có đầy hay không"""
        return len(self.queue) == self.capacity
    
    def enqueue(self, value):
        """Thêm phần tử vào cuối queue nếu queue chưa đầy"""
        if not self.is_full():
            self.queue.append(value)  # Thêm vào cuối list
        else:
            print("Queue đã đầy!")
    
    def dequeue(self):
        """Loại bỏ và trả về phần tử đầu tiên của queue"""
        if not self.is_empty():
            return self.queue.pop(0)  # Lấy và xóa phần tử đầu tiên
        else:
            print("Queue rỗng!")
            return None
    
    def front(self):
        """Lấy giá trị phần tử đầu tiên mà không xóa nó"""
        if not self.is_empty():
            return self.queue[0]  # Trả về phần tử đầu tiên
        else:
            print("Queue rỗng!")
            return None

# Test theo yêu cầu
queue1 = MyQueue(capacity=5)

print("Thêm phần tử vào queue:")
queue1.enqueue(1)
print(f"Phần tử đầu queue sau khi enqueue(1): {queue1.front()}")

queue1.enqueue(2)
print(f"Phần tử đầu queue sau khi enqueue(2): {queue1.front()}")

print(f"\nKiểm tra queue đã đầy chưa: {queue1.is_full()}")  # False

print(f"Phần tử đầu queue: {queue1.front()}")  # 1

print(f"\nLấy ra phần tử đầu queue: {queue1.dequeue()}")  # 1
print(f"Phần tử đầu queue sau khi dequeue: {queue1.front()}")  # 2

print(f"\nLấy ra phần tử đầu queue: {queue1.dequeue()}")  # 2

print(f"\nKiểm tra queue có rỗng không: {queue1.is_empty()}")  # True
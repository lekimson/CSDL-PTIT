# queue_python.py

from collections import deque


class Queue:

    def __init__(self):

        self.elements = deque()  # Sử dụng deque để lưu trữ các phần tử


    def enqueue(self, item):

        self.elements.append(item)  # Thêm phần tử vào cuối deque

        print(f"Đã thêm '{item}' vào hàng đợi.")


    def dequeue(self):

        if not self.is_empty():

            item = self.elements.popleft()  # Loại bỏ phần tử ở đầu deque

            print(f"Đã lấy '{item}' ra khỏi hàng đợi.")

            return item

        else:

            print("Hàng đợi trống!")

            return None


    def front(self):

        if not self.is_empty():

            return self.elements[0]  # Truy cập phần tử đầu tiên

        else:

            print("Hàng đợi trống!")

            return None


    def is_empty(self):

        return len(self.elements) == 0  # Kiểm tra xem deque có trống không


    def size(self):

        return len(self.elements)  # Trả về kích thước hàng đợi


    def display(self):

        print("Hàng đợi (đầu đến cuối):", list(self.elements))  # In hàng đợi từ đầu đến cuối


# Minh họa sử dụng hàng đợi

if __name__ == "__main__":

    queue = Queue()

    queue.enqueue("Tài Liệu 1")

    queue.enqueue("Tài Liệu 2")

    queue.enqueue("Tài Liệu 3")

    queue.display()  # Output: Hàng đợi (đầu đến cuối): ['Tài Liệu 1', 'Tài Liệu 2', 'Tài Liệu 3']


    front_item = queue.front()

    print("Phần tử ở đầu hàng đợi:", front_item)  # Output: Tài Liệu 1


    queue.dequeue()

    queue.display()  # Output: Hàng đợi (đầu đến cuối): ['Tài Liệu 2', 'Tài Liệu 3']


    print("Hàng đợi có trống không?", queue.is_empty())  # Output: False
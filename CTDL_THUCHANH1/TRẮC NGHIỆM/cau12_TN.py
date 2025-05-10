class MyQueue:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__queue = []

    def isEmpty(self):
        return len(self.__queue) == 0

    def is_full(self):
        return len(self.__queue) == self.__capacity

    def dequeue(self):
        if not self.isEmpty():
            return self.__queue.pop(0)
        else:
            print("Queue is empty")

    def enqueue(self, value):
        if not self.is_full():
            self.__queue.append(value)
        else:
            print("Queue is full")

    def front(self):
        # ## Your Code Here
        if not self.isEmpty():
            return self.__queue[0]
        else:
            return None
        # ## End Code Here

# Đoạn kiểm tra
queue1 = MyQueue(capacity=5)
queue1.enqueue(1)
assert queue1.is_full() == False
queue1.enqueue(2)
print(queue1.front())  # => 1

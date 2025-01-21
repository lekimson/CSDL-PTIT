from collections import deque


class RequestQueue:

    def __init__(self):

        self.queue = deque()


    def add_request(self, request):

        self.queue.append(request)

        print(f"Đã thêm yêu cầu '{request}' vào hàng đợi.")


    def process_request(self):

        if self.queue:

            request = self.queue.popleft()

            print(f"Đang xử lý yêu cầu: {request}")

            # Thực hiện xử lý yêu cầu ở đây

        else:

            print("Không có yêu cầu nào để xử lý.")


    def display_queue(self):

        print("Hàng đợi yêu cầu:", list(self.queue))


# Minh họa sử dụng hàng đợi yêu cầu

if __name__ == "__main__":

    rq = RequestQueue()

    rq.add_request("Yêu Cầu 1")

    rq.add_request("Yêu Cầu 2")

    rq.add_request("Yêu Cầu 3")

    rq.display_queue()


    rq.process_request()

    rq.display_queue()


    rq.process_request()

    rq.display_queue()


    rq.process_request()

    rq.display_queue()


    rq.process_request()  # Output: Không có yêu cầu nào để xử lý.
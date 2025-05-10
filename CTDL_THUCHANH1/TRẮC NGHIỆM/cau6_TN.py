from abc import ABC, abstractmethod

# Lớp Person là lớp trừu tượng (abstract class)
class Person(ABC):
    def __init__(self, name: str, yob: int):
        self._name = name
        self._yob = yob
    
    def get_yob(self):
        return self._yob
    
    @abstractmethod
    def describe(self):
        pass

# Lớp Teacher kế thừa từ Person
class Teacher(Person):
    def __init__(self, name: str, yob: int, subject: str):
        super().__init__(name, yob)  # Gọi phương thức khởi tạo của lớp cha (Person)
        self._subject = subject
    
    def describe(self):
        print(f"Teacher - Name: {self._name} - YoB: {self._yob} - Subject: {self._subject}")

# Tạo đối tượng teacher1
teacher1 = Teacher(name="teacherZ2023", yob=1991, subject="History")

# Kiểm tra năm sinh của teacher1
assert teacher1._yob == 1991

# Gọi phương thức describe để in thông tin của teacher1
teacher1.describe()
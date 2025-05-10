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

# Lớp Doctor kế thừa từ Person
class Doctor(Person):
    def __init__(self, name: str, yob: int, specialist: str):
        super().__init__(name, yob)  # Gọi phương thức khởi tạo của lớp cha (Person)
        self._specialist = specialist
    
    def describe(self):
        print(f"Doctor Name: {self._name} - YoB: {self._yob} - Specialist: {self._specialist}")

# Tạo đối tượng doctor1
doctor1 = Doctor(name="doctorZ2023", yob=1981, specialist="Endocrinologists")

# Kiểm tra năm sinh của doctor1
assert doctor1._yob == 1981

# Gọi phương thức describe để in thông tin của doctor1
doctor1.describe()
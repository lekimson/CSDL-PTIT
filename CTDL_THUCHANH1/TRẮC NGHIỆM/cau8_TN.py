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

# Lớp Student kế thừa từ Person
class Student(Person):
    def __init__(self, name: str, yob: int, grade: str):
        super().__init__(name, yob)
        self._grade = grade
    
    def describe(self):
        print(f"Student Name: {self._name} - YoB: {self._yob} - Grade: {self._grade}")

# Lớp Teacher kế thừa từ Person
class Teacher(Person):
    def __init__(self, name: str, yob: int, subject: str):
        super().__init__(name, yob)
        self._subject = subject
    
    def describe(self):
        print(f"Teacher Name: {self._name} - YoB: {self._yob} - Subject: {self._subject}")

# Lớp Doctor kế thừa từ Person
class Doctor(Person):
    def __init__(self, name: str, yob: int, specialist: str):
        super().__init__(name, yob)
        self._specialist = specialist
    
    def describe(self):
        print(f"Doctor Name: {self._name} - YoB: {self._yob} - Specialist: {self._specialist}")

# Lớp Ward để chứa danh sách người và phương thức đếm bác sĩ
class Ward:
    def __init__(self, name: str):
        self._name = name
        self.__listPeople = list()  # Danh sách người trong Ward

    def add_person(self, person: Person):
        """Thêm một người vào danh sách trong Ward"""
        self.__listPeople.append(person)
    
    def describe(self):
        """In thông tin Ward và các người trong đó"""
        print(f"Ward Name: {self._name}")
        for p in self.__listPeople:
            p.describe()
    
    def count_doctor(self):
        """Đếm số lượng Doctor trong Ward"""
        count = 0
        for p in self.__listPeople:
            if isinstance(p, Doctor):  # Kiểm tra nếu đối tượng là Doctor
                count += 1
        return count

# Tạo các đối tượng
student1 = Student(name="studentA", yob=2010, grade="7")
teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")
doctor2 = Doctor(name="doctorB", yob=1975, specialist="Cardiologists")

# Tạo đối tượng Ward và thêm các người vào
ward1 = Ward(name="Ward1")
ward1.add_person(student1)
ward1.add_person(teacher1)
ward1.add_person(teacher2)
ward1.add_person(doctor1)
ward1.add_person(doctor2)

# Đếm số lượng Doctor trong Ward
print(ward1.count_doctor())  # In ra kết quả đếm số lượng Doctor
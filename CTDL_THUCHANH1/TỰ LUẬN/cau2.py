from abc import ABC, abstractmethod

# Lớp trừu tượng Person
class Person(ABC):
    def __init__(self, name: str, yob: int):
        self._name = name
        self._yob = yob

    def get_yob(self):
        return self._yob

    @abstractmethod
    def describe(self):
        pass

# Lớp Student kế thừa Person
class Student(Person):
    def __init__(self, name: str, yob: int, grade: str):
        super().__init__(name, yob)
        self._grade = grade

    def describe(self):
        print(f"Student - Name: {self._name} - YoB: {self._yob} - Grade: {self._grade}")

# Lớp Teacher kế thừa Person
class Teacher(Person):
    def __init__(self, name: str, yob: int, subject: str):
        super().__init__(name, yob)
        self._subject = subject

    def describe(self):
        print(f"Teacher - Name: {self._name} - YoB: {self._yob} - Subject: {self._subject}")

# Lớp Doctor kế thừa Person
class Doctor(Person):
    def __init__(self, name: str, yob: int, specialist: str):
        super().__init__(name, yob)
        self._specialist = specialist

    def describe(self):
        print(f"Doctor - Name: {self._name} - YoB: {self._yob} - Specialist: {self._specialist}")

# Lớp Ward để quản lý danh sách người
class Ward:
    def __init__(self, name: str):
        self.__name = name
        self.__listPeople = []

    def add_person(self, person: Person):
        self.__listPeople.append(person)

    def describe(self):
        print(f"Ward Name: {self.__name}")
        for p in self.__listPeople:
            p.describe()

    def count_doctor(self):
        return sum(1 for p in self.__listPeople if isinstance(p, Doctor))

    def sort_age(self):
        self.__listPeople.sort(key=lambda p: p.get_yob(), reverse=True)

    def compute_average(self):
        teachers = [p.get_yob() for p in self.__listPeople if isinstance(p, Teacher)]
        return sum(teachers) / len(teachers) if teachers else 0

# Kiểm thử các chức năng

# (a) Tạo đối tượng và mô tả
student1 = Student(name="studentA", yob=2010, grade="7")
student1.describe()

teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
teacher1.describe()

doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")
doctor1.describe()

# (b) Thêm vào Ward và hiển thị danh sách
print()
teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
doctor2 = Doctor(name="doctorB", yob=1975, specialist="Cardiologists")
ward1 = Ward(name="Ward1")
ward1.add_person(student1)
ward1.add_person(teacher1)
ward1.add_person(teacher2)
ward1.add_person(doctor1)
ward1.add_person(doctor2)
ward1.describe()

# (c) Đếm số bác sĩ
print(f"\nNumber of doctors: {ward1.count_doctor()}")

# (d) Sắp xếp theo tuổi (năm sinh giảm dần)
print("\nAfter sorting Age of Ward1 people")
ward1.sort_age()
ward1.describe()

# (e) Tính năm sinh trung bình của giáo viên
print(f"\nAverage year of birth (teachers): {ward1.compute_average():.1f}")
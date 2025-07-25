from pydantic import BaseModel
from typing import Optional

class student(BaseModel):
    name:str
    age:Optional[int]=None

new_student1={'name':'Kamal'}
student1=student(**new_student1)
print(type(student1))

# new_student2={'name':32,'age':25}
# student2=student(**new_student2)
# print(type(student2))

new_student3={'name':'Kamal','age':25}
student3=student(**new_student3)
print(type(student3))

new_student4={'name':'Kamal'}
student4=student(**new_student4)
print(type(student4))





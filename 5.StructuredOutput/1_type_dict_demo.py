from typing import TypedDict

class Employee(TypedDict):
    id:int
    name:str
    salary:float

new_person:Employee={'id':45,'name':"ksmsl",'salary':0.0}
print(new_person)
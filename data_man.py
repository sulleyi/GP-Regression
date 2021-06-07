import pandas as pd
import numpy as np
import csv


''' HOW THEY IMPOT DATA IN THE EXAMPLE GP PROGRAM
with open("data.csv") as database:
    dataReader = csv.reader(database)
    data = list(list(float(elem) for elem in row) for row in dataReader)
'''
class_to_num = {
    "B" : 0,
    "M" : 1
    }

def class_to_num(row):
    if row == "M":
        return 1
    else:
        return 0

breastcancer = pd.read_csv('data.csv')
breastcancer = breastcancer.iloc[:, 1:31]
breastcancer['diagnosis'] = breastcancer['diagnosis'].apply(class_to_num)

print("breastcancer info: ")
print(breastcancer.info())


diagnosis = breastcancer['diagnosis']
print("diagnosis")
print(diagnosis)


cancer_xs = breastcancer.iloc[:, 1:]
print("cancer xs: ")
print(cancer_xs)

print("cancer y: ")
print(breastcancer["diagnosis"])


for col in breastcancer:
    print(col)
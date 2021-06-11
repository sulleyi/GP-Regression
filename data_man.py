import pandas as pd
import numpy as np
import csv


''' HOW THEY IMPOT DATA IN THE EXAMPLE GP PROGRAM


with open("data.csv") as database:
    dataReader = csv.reader(database)
    data = list(list(elem for elem in row) for row in dataReader)

for row in data:
    row = row[1:]
    #print(row)
    if row[0] == "M":
        row[0] = True
        
        
    else:
        row[0] = False
        
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


breastcancer_copy = breastcancer.copy()
train_w_replacement = breastcancer_copy.sample(frac=.8, replace=True, random_state=55)
test_w_replacement = breastcancer_copy.drop(train_w_replacement.index)

breastcancer_copy = breastcancer.copy()
train_wo_replacement = breastcancer_copy.sample(frac=.8, replace=False, random_state=55)
test_wo_replacement = breastcancer_copy.drop(train_w_replacement.index)


'''
def getColumnNames(df):

    argsdict ={}
    args = []
    for i in range(df.shape[1]):
        print(df.columns[i])
        argsdict["ARG{0}".format(i)] = df.columns[i]
        args.append(df.columns[i])
    print(argsdict)
    return argsdict


args = getColumnNames(cancer_xs)
'''
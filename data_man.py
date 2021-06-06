import pandas as pd
import numpy as np


'''
IMPORT DATASET TODO
'''
diamonds = pd.read_csv('Diamond_Data.csv')
print(diamonds)
diamonds.dropna
dummydiamonds = pd.get_dummies(diamonds)

diamond_y_price = diamonds.iloc[:, np.r_[0]]
print(diamond_y_price)
diamond_xs = diamonds.iloc[:, np.r_[1:7]]
print(diamond_xs)
dummydiamond_xs = pd.get_dummies(diamond_xs)

def getColumnNames(df):
    
    argsdict ={}
    args = []
    for i in range(df.shape[1]):
        print(df.columns[i])
        argsdict["ARG{0}".format(i)] = df.columns[i]
        args.append(df.columns[i])
    print(argsdict)
    return args


args = getColumnNames(dummydiamond_xs)
print(args)

points = []
for i in range(dummydiamond_xs.shape[1]):
    points.append(list(dummydiamond_xs.iloc[:, np.r_[i]]
))




#points = tuple(dummydiamond_xs[:, i:i+1] for i in range(dummydiamond_xs.shape[1] -1))
print("points =")
print(points)
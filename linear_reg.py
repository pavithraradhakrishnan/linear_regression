
import numpy as np
import csv
import math

from operator import itemgetter
x_train =[]
y_train = []
test_data =[]
def data_clean(x_train,y_train):
    with open("input_data.csv",encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile,delimiter=',')
        data = [data for data in rows]


    for x in range(len(data)):
        ins = []

        x_train.append(float(data[x][0]))
        y_train.append(float(data[x][1]))



    print(np.array(x_train))
    print(np.array(y_train))
    x = np.array(x_train)
    y = np.array(y_train)
    print("x in data clean", x)
    print("y in data clean",y)

    x_train_mean = np.mean(x)
    print("x_mean",x_train_mean)
    y_train_mean = np.mean(y)
    m,c= 0,0
    sum = 0
    numerator,denominator = 0,0
    for i in range(len(x)):
        numerator += (x[i] - x_train_mean) * (y[i] - y_train_mean)
        denominator += (x[i] - x_train_mean) ** 2
    m = numerator/denominator
    c = y_train_mean - (m) * (x_train_mean)
    return x,y,x_train_mean,y_train_mean,m,c
def calculate_squared_error(x,y,x_train_mean,y_train_mean,m,c):
    nr,dr = 0,0
    for i in range(len(x)):
        y_pred = m * x[i] + c
        nr += (y[i] - y_train_mean) ** 2
        dr += (y[i] - y_pred ) ** 2
        rmse = 1 - (nr/dr)
    return rmse






x,y,x_train_mean,y_train_mean,m,c = data_clean([],[])
rmse = calculate_squared_error(x,y,x_train_mean,y_train_mean,m,c)
print("The bias coefficients are ",m,c)
print("the root mean squared error is",rmse)










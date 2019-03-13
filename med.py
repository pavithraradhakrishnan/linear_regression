import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from operator import itemgetter
x_train =[]
y_train = []
test_data =[]
data_set =[]
def data_clean(x_train,y_train,test_data):
    with open("new.csv",encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile,delimiter=',')
        data = [data for data in rows]


    for l in range(len(data)):
        ins = []
        for m in range(2):
            ins.append(float(data[l][m]))
        x_train.append(ins)
    for k in range(len(data)):
        y_train.append(float(data[k][2]))

    for l in range(len(data)):
        ins = []
        for m in range(3):
            ins.append(float(data[l][m]))
        data_set.append(ins)
    with open("2d_output.csv",encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile,delimiter=',')
        dota = [dota for dota in rows]



    for k in range(len(dota)):
        ins = []
        for b in range(3):
            ins.append(float(dota[k][b]))
        test_data.append(ins)



    #print(np.array(x_train))
    #print(np.array(y_train))
    test = np.array(test_data)

    x = np.array(x_train)
    y = np.array(y_train)
    data = np.array(data_set)
    test= np.array(test_data)
    #print("x in data clean",x)
    #print("y in data clean",y)
    #print("test in data clean",test)
    #print("data set",data)
    #x_train = x
    #y_train = y
    return x,y,data,test

def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))

    h = h.reshape(X.shape[0])
    return h

def gradient_descent(theta, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    cost1 = np.ones(num_iters)
    cost2 = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/X.shape[0]) * sum((h-y) * X.transpose()[j])
        h = hypothesis(theta, X, n)
        h1 = hypothesis(theta, X*X, n)

        h2 = hypothesis(theta, X*X*X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
        #cost1[i] = (1 / X.shape[0]) * 0.5 * sum(np.square(h - y))
        #cost2[i] = (1 / X.shape[0]) * 0.5 * sum(np.square(h - y))
    #theta = theta.reshape(1,n)
    return theta, cost


def linear_regression(X, y, alpha, num_iters):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    #print("cap x is",X)
    theta = np.zeros(n +1)

    h = hypothesis(theta, X, n)

    theta, cost= gradient_descent(theta,alpha,num_iters,h,X,y,n)
    return theta, cost,X,h



def predictPrice(x, theta):
        return np.dot(x, theta)



x,y,data,test = data_clean([],[],[])

theta, cost,X,h = linear_regression(x,y,0.0001, 100)
#print("theta",theta)
#print("cost",cost)
n = len(test)
#theta = np.zeros(n+1)

y_predict = predictPrice(X,theta)
print("THe predicted price is ",y_predict)
error = y_predict-y
error_first =(1/(2*len(y)))*np.sum(error**2)
print(" the regression coefficients for frist order",theta)
print("the cost for the first order",min(cost))
print("error of first order", error_first)
#h = hypothesis(theta,X,len(x))
#print("hypothesis value is", h)
sq_X = np.square(X)
theta1, cost1,sq_X,h= linear_regression(sq_X,y,0.0001, 100)
#y_predict1 = predictPrice(X,theta1)
#print("THe predicted price is ",y_predict)
#error = y_predict1 -y
#error_second =(1/(2*len(y)))*np.sum(error**2)


#print("theta",theta1)
#print("cost",cost1)
#print("X",sq_X)

print(" the regression coefficients for second order",theta1)
print("the cost for the second order",min(cost1))
#print("error of second order is ",error_second)
#th_X = np.square(sq_X)
#theta2, cost2,th_X= linear_regression(th_X,y,0.0001, 100)
#print("theta",theta2)
#print("cost",cost2)
#print("X",th_X)
#print(" the regression coefficients for fourth order",theta2)
#print("the cost for the fourth order",min(cost2))






#print("the cost1 is ",min(cost1))
#print("the cost2 is ",min(cost2))
#print("the predictions are",predict)






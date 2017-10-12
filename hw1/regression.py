import csv 
import numpy as np
import sys

input_url = sys.argv[1]
output_url = sys.argv[2]

# Model: Y = Bias + W1*X1 + W2X2....+Wn*Xn 

# AMB_TEMP=0, CH4=1, CO=2, NMHC=3, NO=4, NO2 = 5
# NOx=6, O3=7, PM10 =8, PM2.5=9, RAIN_FALL = 10, RH=11
# SO2 = 12, THC=13, WD_HR=14, WIND_DIREC=15, WS = 16, WS_HR = 17
training_day = 20
iteration = 1000000

feature_hour = 9
feature_measure = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
add_square = False
add_cube = False
lambdaa = 0
stop = 0.001

"""
data = []
for i in range(18):
    data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5')
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        for i in range(3,27):
            if r[i] == "NR":
                data[(n_row-1)%18].append(float(0))
            else:
                data[(n_row-1)%18].append(float(r[i]))
    n_row = n_row+1
text.close()

x = []
y = []
length_a_month = training_day*24 - feature_hour
for month in range(12):
    for hour in range(length_a_month):
        tmp = []
        for row in range(len(feature_measure)):
            for col in range(feature_hour):
                tmp.append(data[feature_measure[row]][480*month+hour+col])
        x.append(tmp)
        y.append(data[9][480*month+hour+feature_hour]) # PM2.5

x = np.array(x)
y = np.array(y)

#add square term
if add_square == True:
	x = np.concatenate((x,x**2), axis=1)
if add_cube == True:
	x = np.concatenate((x,x**3), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
lr = 10
pre_cost = 1000000

x_t = x.transpose()
s_grad = np.zeros(len(x[0]))

for i in range(iteration):
    dist = np.dot(x,w) - y
    loss = sum(dist**2)
    cost = np.sqrt(loss / len(x))
    grad = np.dot(x_t,dist) * 2 + 2 * lambdaa  * w
    s_grad += grad**2
    ada = np.sqrt(s_grad)
    w = w - lr * grad/ada
    if i%100 == 0:   
        print ("\033[94miteration: \033[95m%d  \033[92mCost: \033[95m%f "%(i,cost))
        if (pre_cost-cost) < stop:
            break
        pre_cost = cost

# save model
np.save("./model/model.npy",w)
"""



#-----------------#
#-----Testing-----#
#-----------------#
# read model
w = np.load("./model/model.npy")
test_x = []
n_row = 0
text = open(input_url, "r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row%18 == 0:
        test_x.append([])
        for n in range(len(feature_measure)):
            if n_row%18 == feature_measure[n]:
                for i in range(11-feature_hour,11):
                    test_x[n_row//18].append(float(r[i]))
                break
    else :
        for n in range(len(feature_measure)):
            if n_row%18 == feature_measure[n]:
                for i in range(11-feature_hour,11):
                    if r[i] !="NR":
                        test_x[n_row//18].append(float(r[i]))
                    else:
                        test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
if add_square == True:
	test_x = np.concatenate((test_x,test_x**2), axis=1)
if add_cube == True:
	test_x = np.concatenate((test_x,test_x**3), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

pre = []
for i in range(len(test_x)):
    pre.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    pre[i].append(a)

text = open(output_url, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(pre)):
    s.writerow(pre[i])
text.close()
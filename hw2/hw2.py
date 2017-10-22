import csv 
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


#-------------#
#-Initialize--#
#-------------# 
# num_validation = 10853
num_validation = 16280
num_training = 32561 - num_validation

lambdaa = 0
feature = [0,1,2,3,4,5, 
			6,7,8,9,10,11,12,13,14, 
			15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
			31,32,33,34,35,36,37,
			38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,
			53,54,55,56,57,58,
			59,60,61,62,63,
			64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,
			85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105]
add_squre = True

stop = 0.01
lr = 1e-8
iteration = 100000
"""
feature = [0,1,2,3,4,5, 
			6,7,8,9,10,11,12,13,14, 
			15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
			31,32,33,34,35,36,37,
			38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,
			53,54,55,56,57,58,
			59,60,61,62,63,
			64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,
			85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105]
"""


#------------#
#-Read Data--#
#------------#
# x
X = []
val_X = []
n_row = 0
text = open('data/X_train', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row == 0:
		pass
	elif n_row <= num_training:
		tmp = []
		for i in range(len(feature)):
			tmp.append(float(r[feature[i]]))
		X.append(tmp)
	else: 
		tmp = []
		for i in range(len(feature)):
			tmp.append(float(r[feature[i]]))
		val_X.append(tmp)
	n_row = n_row + 1
text.close()
X = np.array(X)
val_X = np.array(val_X)
# normalization
for i in range(0, X.shape[1], 1):
	mean = np.mean(X[:,i])
	std = np.std(X[:,i])
	if std != 0:
		X[:,i] = (X[:,i] - mean) / std
for i in range(0, val_X.shape[1], 1):
	mean = np.mean(val_X[:,i])
	std = np.std(val_X[:,i])
	if std != 0:
		val_X[:,i] = (val_X[:,i] - mean) / std

# add square term
if add_squre == True:
	X = np.concatenate((X,X**2), axis=1)
if add_squre == True:
	val_X = np.concatenate((val_X,val_X**2), axis=1)
# add bias
X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)
val_X = np.concatenate((np.ones((val_X.shape[0],1)),val_X), axis=1)

# y
Y = []
val_Y = []
n_row = 0
text = open('data/Y_train', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row == 0:
		pass
	elif n_row <= num_training:
		Y.append(float(r[0]))
	else:
		val_Y.append(float(r[0]))
	n_row = n_row + 1
text.close()
Y = np.array(Y)
val_Y = np.array(val_Y)

#------------#
#-Training1--#
#------------#
w = np.zeros(len(X[0]))

X_t = X.transpose()
s_gra = np.zeros(len(X[0]))
old_cost = 100000000
for i in range(iteration):
	z = np.dot(X,w)
	y = sigmoid(z)
	cost = -(np.dot(Y,np.log(y)) + np.dot((1-Y),np.log(1-y)))
	loss = y - Y
	gra = np.dot(X_t,loss)*2 + 2*lambdaa*w
	w = w - lr * gra
	
	if i % 100 == 0:
		print ('iteration: %d | Cost: %f  ' % ( i,cost))
		if (old_cost - cost) < stop:
			break
		old_cost = cost


#--------------#
#-Validation1--#
#--------------#
hit = 0
for i in range(len(val_X)):
	a = sigmoid(np.dot(w,val_X[i]))
	if a < 0.5 and val_Y[i] == 0:
		hit = hit + 1
	elif a >= 0.5 and val_Y[i] == 1:
		hit = hit + 1

hit_rate1 = hit / len(val_Y)


#------------#
#-Training2--#
#------------#
tmp = X
X = val_X
val_X = tmp

tmp = Y
Y = val_Y
val_Y = tmp

w = np.zeros(len(X[0]))
X_t = X.transpose()
s_gra = np.zeros(len(X[0]))
old_cost = 100000000
for i in range(iteration):
	z = np.dot(X,w)
	y = sigmoid(z)
	cost = -(np.dot(Y,np.log(y)) + np.dot((1-Y),np.log(1-y)))
	loss = y - Y
	gra = np.dot(X_t,loss)*2 + 2*lambdaa*w
	w = w - lr * gra
	if i % 100 == 0:
		print ('iteration: %d | Cost: %f  ' % ( i,cost))
		if (old_cost - cost) < stop:
			break
		old_cost = cost


#--------------#
#-Validation2--#
#--------------#
hit = 0
for i in range(len(val_X)):
	a = sigmoid(np.dot(w,val_X[i]))
	if a < 0.5 and val_Y[i] == 0:
		hit = hit + 1
	elif a >= 0.5 and val_Y[i] == 1:
		hit = hit + 1
print("------------------------")
hit_rate2 = hit / len(val_Y)
print("accuracy1:",hit_rate1)
print("accuracy2:",hit_rate2)
print("Avg acc:",(hit_rate1+hit_rate2)/2)



#--------------------#
#-Reading Test data--#
#--------------------#
test_X = []
n_row = 0
text = open('data/X_test' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
	if n_row == 0:
		pass
	else:
		tmp = []
		for i in range(len(feature)):
			tmp.append(float(r[feature[i]]))
		test_X.append(tmp)
	n_row = n_row+1
text.close()
test_X = np.array(test_X)

# normalization
#test_x = (test_x - test_x.min(0)) / (test_x.max(0) - test_x.min(0))
for i in range(0, test_X.shape[1], 1):
	mean = np.mean(test_X[:,i])
	std = np.std(test_X[:,i])
	if std != 0:
		test_X[:,i] = (test_X[:,i] - mean) / std

# add square term
if add_squre == True:
	test_X = np.concatenate((test_X,test_X**2), axis=1)
# add bias
test_X = np.concatenate((np.ones((test_X.shape[0],1)),test_X), axis=1)

ans = []
for i in range(len(test_X)):
	ans.append([str(i+1)])
	a = sigmoid(np.dot(w,test_X[i]))
	if a <= 0.5:
		ans[i].append('0')
	else:
		ans[i].append('1')

filename = "result/predict.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i])
text.close()

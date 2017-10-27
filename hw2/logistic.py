import csv 
import numpy as np
import sys

test_url = sys.argv[1]
output_url =  sys.argv[2]

# global variable
hit_rate1 = 0
hit_rate2 = 0

def sigmoid(x):
	res =  1 / (1.0 + np.exp(-x))
	return np.clip(res,1e-8, 1-(1e-8))
def normalize(X):
	for i in range(0, X.shape[1], 1):
		mean = np.mean(X[:,i])
		std = np.std(X[:,i])
		if std != 0:
			X[:,i] = (X[:,i] - mean) / std
		else:
			print("std is zero:",i,X[:,i])
	return X
def training_and_validation(X,val_X,Y,val_Y,type):
	global hit_rate1, hit_rate2
	w = np.zeros(len(X[0]))
	X_t = X.transpose()
	for i in range(iteration):
		z = np.dot(X,w)
		y = sigmoid(z)
		loss = y - Y
		gra = np.dot(X_t,loss)*2 + 2*lambdaa*w
		w = w - lr * gra
		if i % 100 == 0:
			cost = -(np.dot(Y,np.log(y)) + np.dot((1-Y),np.log(1-y)))
			print ('iteration: %d | Cost: %f  ' % ( i,cost/len(X)))
	#--------------#
	#--Validation--#
	#--------------#
	hit = 0
	for i in range(len(val_X)):
		a = sigmoid(np.dot(w,val_X[i]))
		if a < 0.5 and val_Y[i] == 0:
			hit = hit + 1
		elif a >= 0.5 and val_Y[i] == 1:
			hit = hit + 1
	if type == 1:
		hit_rate1 = hit / len(val_Y)
	elif type == 2:
		hit_rate2 = hit / len(val_Y)

#--------------#
#--Initialize--#
#--------------# 
num_validation = 16280
num_training = 32561 - num_validation

lambdaa = 1
feature = [0,1,2,3,4,5, 
			6,7,8,9,10,11,12,13,14, 
			15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
			31,32,33,34,35,36,37,
			38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,
			53,54,55,56,57,58,
			59,60,61,62,63,
			64,65,66,67,68,69,70,71,72,73,74,75,76,77,79,80,81,82,83,84,
			85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105]
add_squre = True
add_cube = True
lr = 1e-5
iteration = 10000
#stop = 0.00000000001
train_all_data = True
"""
print("iteration:",iteration)
print("lambda:",lambdaa)

#-------------#
#--Read Data--#
#-------------#
# x
X_1 = []
X_2 = []
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
		X_1.append(tmp)
	else: 
		tmp = []
		for i in range(len(feature)):
			tmp.append(float(r[feature[i]]))
		X_2.append(tmp)
	n_row = n_row + 1
text.close()
X_1 = np.array(X_1)
X_2 = np.array(X_2)

# add square term
if add_squre == True:
	X_1 = np.concatenate((X_1,X_1**2), axis=1)
	X_2 = np.concatenate((X_2,X_2**2), axis=1)
# add cube term
if add_cube == True:
	X_1 = np.concatenate((X_1,X_1**3), axis=1)
	X_2 = np.concatenate((X_2,X_2**3), axis=1)
# X
X = np.concatenate((X_1,X_2), axis=0)
# normalization
X_1 = normalize(X_1)
X_2 = normalize(X_2)
X = normalize(X)
# add bias
X_1 = np.concatenate((np.ones((X_1.shape[0],1)),X_1), axis=1)
X_2 = np.concatenate((np.ones((X_2.shape[0],1)),X_2), axis=1)
X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)

# y
Y_1 = []
Y_2 = []
Y = []
n_row = 0
text = open('data/Y_train', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row == 0:
		pass
	elif n_row <= num_training:
		Y_1.append(float(r[0]))
	else:
		Y_2.append(float(r[0]))
	n_row = n_row + 1
text.close()
Y_1 = np.array(Y_1)
Y_2 = np.array(Y_2)
Y = np.concatenate((Y_1,Y_2), axis=0)

#-------------------------#
#--Training & Validation--#
#-------------------------#
if train_all_data == False:
	print("Training....")
	training_and_validation(X_1,X_2,Y_1,Y_2,1)
	training_and_validation(X_2,X_1,Y_2,Y_1,2)
	print("--------------Done!-------------------")
	print("accuracy1:",hit_rate1)
	print("accuracy2:",hit_rate2)
	print("Avg acc:",(hit_rate1+hit_rate2)/2)

#----------#
#--Output--#
#----------#
#--Training--
if train_all_data == True:
	print("Training....")
	w = np.zeros(len(X[0]))
	X_t = X.transpose()
	for i in range(iteration):
		z = np.dot(X,w)
		y = sigmoid(z)
		loss = y - Y
		gra = np.dot(X_t,loss)*2 + 2*lambdaa*w
		w = w - lr * gra
			
		if i % 100 == 0:
			cost = -(np.dot(Y,np.log(y)) + np.dot((1-Y),np.log(1-y)))
			print ('iteration: %d | Cost: %f  ' % ( i,cost/len(X)))
	# save model
	np.save("./model/model_best.npy",w)

	# Test
	hit = 0
	for i in range(len(Y)):
		if y[i] < 0.5 and Y[i] == 0:
			hit = hit + 1
		elif y[i] >= 0.5 and Y[i] == 1:
			hit = hit + 1
	print("Acc in training: ",hit/len(Y))
"""
w = np.load("./model/model_best.npy")
test_X = []
n_row = 0
text = open(test_url ,"r")
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
# add square term
if add_squre == True:
	test_X = np.concatenate((test_X,test_X**2), axis=1)
# add cube term
if add_cube == True:
	test_X = np.concatenate((test_X,test_X**3), axis=1)
# normalization
for i in range(0, test_X.shape[1], 1):
	mean = np.mean(test_X[:,i])
	std = np.std(test_X[:,i])
	if std != 0:
		test_X[:,i] = (test_X[:,i] - mean) / std
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
	text = open(output_url, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i])
text.close()
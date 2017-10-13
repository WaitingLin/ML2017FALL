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
feature_measure = [8,9,12]
add_square = True
add_cube = False
lambdaa = 0
stop = 0.0001
b1 = 0.5
b2 = 0.5

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
#-----------------#
#----Training-1---#
#-----------------#
x = []
y = []
length_a_month = training_day*24 - feature_hour
for month in range(12):
	for hour in range(length_a_month):
		delet = False
		tmp = []
		for row in range(len(feature_measure)):
			for col in range(feature_hour):
				if data[feature_measure[row]][480*month+hour+col] == -1:
					delet = True
				tmp.append(data[feature_measure[row]][480*month+hour+col])
		if delet == False:
			x.append(tmp)
			y.append(data[9][480*month+hour+feature_hour]) # PM2.5
		else:
			pass
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
mom = np.zeros(len(x[0]))
ada = np.zeros(len(x[0]))

iteration1 = iteration
for i in range(iteration):
	dist = np.dot(x,w) - y
	loss = sum(dist**2)
	cost = np.sqrt(loss / len(x))
	grad = np.dot(x_t,dist) * 2 + 2 * lambdaa  * w 
	s_grad += grad**2
	mom = b1 * mom + (1-b1) * grad
	ada = b2 * ada + (1-b2) * np.sqrt(s_grad)
	w = w - lr * mom/ada 
	if i%100 == 0:
		print ("\033[94miteration: \033[95m%d  \033[92mCost: \033[95m%f "%(i,cost))
		if abs(pre_cost-cost) < stop:
			iteration1 = i
			break
		pre_cost = cost
cost1 = cost
# save model
np.save("./model/model_best.npy",w)

#-----------------#
#---Validation-1--#
#-----------------#
if training_day != 20:
	v = []
	ans = []
	length_a_month = (20-training_day)*24 - feature_hour
	for month in range(12):
		for hour in range(length_a_month):
			delet = False
			tmp = []
			for row in range(len(feature_measure)):
				for col in range(feature_hour):
					if data[feature_measure[row]][480*month+training_day*24+hour+col] == -1:
						delet = True
					tmp.append(data[feature_measure[row]][480*month+training_day*24+hour+col])
			if delet == False:
				v.append(tmp)
				ans.append(data[9][480*month+training_day*24+hour+feature_hour]) # PM2.5
			else:
				pass
	v = np.array(v)
	ans = np.array(ans)
	#add square term
	if add_square == True:
		v = np.concatenate((v,v**2), axis=1)
	if add_cube == True:
		v = np.concatenate((v,v**3), axis=1)
	# add bias
	v = np.concatenate((np.ones((v.shape[0],1)),v), axis=1)
	pre = []
	for i in range(len(v)):
		a = np.dot(w,v[i])
		pre.append(a)
	# Cost
	error1 = np.sqrt(sum((pre - ans)**2) / len(pre))
#-----------------#
#----Training-2---#
#-----------------#
	x = []
	y = []
	length_a_month = (20-training_day)*24 - feature_hour
	for month in range(12):
		for hour in range(length_a_month):
			delet = False
			tmp = []
			for row in range(len(feature_measure)):
				for col in range(feature_hour):
					if data[feature_measure[row]][480*month+training_day*24+hour+col] == -1:
						delet = True
					tmp.append(data[feature_measure[row]][480*month+training_day*24+hour+col])
			if delet == False:
				x.append(tmp)
				y.append(data[9][480*month+training_day*24+hour+feature_hour]) # PM2.5
			else:
				pass
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
	mom = np.zeros(len(x[0]))
	ada = np.zeros(len(x[0]))
	
	iteration2 = iteration
	for i in range(iteration):
		dist = np.dot(x,w) - y
		loss = sum(dist**2)
		cost = np.sqrt(loss / len(x))
		grad = np.dot(x_t,dist) * 2 + 2 * lambdaa  * w
		s_grad += grad**2
		mom = b1 * mom + (1-b1) * grad
		ada = b2 * ada + (1-b2) * np.sqrt(s_grad)
		w = w - lr * mom/ada 
		if i%100 == 0:   
			print ("\033[94miteration: \033[95m%d  \033[92mCost: \033[95m%f "%(i,cost))
			if abs(pre_cost-cost) < stop:
				iteration2 = i
				break
			pre_cost = cost
	cost2 = cost
	#-----------------#
	#---Validation-2--#
	#-----------------#
	v = []
	ans = []
	length_a_month = training_day*24 - feature_hour
	for month in range(12):
		for hour in range(length_a_month):
			delet = False
			tmp = []
			for row in range(len(feature_measure)):
				for col in range(feature_hour):
					if data[feature_measure[row]][480*month+hour+col] == -1:
						delet = True
					tmp.append(data[feature_measure[row]][480*month+hour+col])
			if delet == False:
				v.append(tmp)
				ans.append(data[9][480*month+hour+feature_hour]) # PM2.5
			else:
				pass
	v = np.array(v)
	ans = np.array(ans)
	#add square term
	if add_square == True:
		v = np.concatenate((v,v**2), axis=1)
	if add_cube == True:
		v = np.concatenate((v,v**3), axis=1)
	# add bias
	v = np.concatenate((np.ones((v.shape[0],1)),v), axis=1)
	pre = []
	for i in range(len(v)):
		a = np.dot(w,v[i])
		pre.append(a)
	error2 = np.sqrt(sum((pre - ans)**2) / len(pre))
	#---------#
	#--Print--#
	#---------#
	print("---------------------------------")
	print("\033[94miteration: \033[95m%d  \033[92mCost: \033[95m%f \033[94mError:\033[95m%f"%(iteration1,cost1,error1))
	print("\033[94miteration: \033[95m%d  \033[92mCost: \033[95m%f \033[94mError:\033[95m%f"%(iteration2,cost2,error2))
	print("\033[94mAvg Err: \033[95m%f"%((error1+error2)/2))
"""

#-----------------#
#-----Testing-----#
#-----------------#
# read model
w = np.load("./model/model_best.npy")

test_x = []
n_row = 0
text = open(input_url ,"r")
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
						#test_x[n_row//18].append(float(r[i]))
						if r[i] == '-1':
							if i == 2:
								test_x[n_row//18].append(float(r[i+1]))
							elif i == 10:
								test_x[n_row//18].append(float(r[i-1]))
							else:
								test_x[n_row//18].append((float(r[i+1])+float(r[i-1]))/2)
						else:
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
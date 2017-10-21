import csv 
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#------------#
#-read data--#
#------------#
x = []
n_row = 0
text = open('data/X_train', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row == 0:
		pass
	else:
		tmp = []
		for i in range(106):
			tmp.append(float(r[i]))
		x.append(tmp)
	n_row = n_row + 1
text.close()
x = np.array(x)

# normalization
#x = (x - x.min(0)) / (x.max(0) - x.min(0))
for i in range(0, x.shape[1], 1):
	mean_val = np.mean(x[:,i])
	std_val = np.std(x[:,i])
	if std_val != 0:
		x[:,i] = (x[:,i] - mean_val) / std_val

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

y = []
n_row = 0
text = open('data/Y_train', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row == 0:
		pass
	else:
		y.append(float(r[0]))
	n_row = n_row + 1
text.close()
y = np.array(y)

#-----------#
#-Training--#
#-----------#
w = np.zeros(len(x[0]))
lr = 5e-6
iteration = 1000

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(iteration):
	z = np.dot(x,w)
	hypo = sigmoid(z)
	loss = hypo - y
	cost = np.sum(0 - y*np.log(hypo) - (1-y)*np.log(1-hypo))
	gra = np.dot(x_t,loss)
	# s_gra += gra**2
	# ada = np.sqrt(s_gra)
	# w = w - lr * gra/ada
	w = w - lr * gra
	print ('iteration: %d | Cost: %f  ' % ( i,cost))

hit = 0
for i in range(len(y)):
	if hypo[i] > 0.5 and y[i]==1:
		hit = hit + 1
	elif hypo[i] < 0.5 and y[i]==0:
		hit = hit + 1
hitrate = hit / len(y)
print("accuracy:",hitrate)

#--------------------#
#-Reading Test data--#
#--------------------#
test_x = []
n_row = 0
text = open('data/X_test' ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
	if n_row == 0:
		pass
	else:
		tmp = []
		for i in range(106):
			tmp.append(float(r[i]))
		test_x.append(tmp)
	n_row = n_row+1
text.close()
test_x = np.array(test_x)

# normalization
#test_x = (test_x - test_x.min(0)) / (test_x.max(0) - test_x.min(0))
for i in range(0, test_x.shape[1], 1):
	mean_val = np.mean(test_x[:,i])
	std_val = np.std(test_x[:,i])
	if std_val != 0:
		test_x[:,i] = (test_x[:,i] - mean_val) / std_val

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
	ans.append([str(i+1)])
	a = sigmoid(np.dot(w,test_x[i]))
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
# push test 2 

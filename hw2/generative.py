import csv 
import numpy as np
import sys

test_url = sys.argv[1]
output_url =  sys.argv[2]

def sigmoid(x):
	res =  1 / (1.0 + np.exp(-x))
	return np.clip(res,1e-8, 1-(1e-8))

"""
#-------------#
#--Read Data--#
#-------------#
# x
X = []
n_row = 0
text = open('./hw2_data/X_train', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row == 0:
		pass
	else: 
		tmp = []
		for i in range(106):
			tmp.append(float(r[i]))
		X.append(tmp)
	n_row = n_row + 1
text.close()
# y
Y = []
n_row = 0
text = open('./hw2_data/Y_train', 'r') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row == 0:
		pass
	else:
		Y.append(float(r[0]))
	n_row = n_row + 1
text.close()
X = np.array(X)
Y = np.array(Y)

# Gaussian distribution parameters
train_data_size = X.shape[0]
cnt1 = 0
cnt2 = 0

mu1 = np.zeros((106,))
mu2 = np.zeros((106,))
for i in range(train_data_size):
	if Y[i] == 1:
		mu1 += X[i]
		cnt1 += 1
	else:
		mu2 += X[i]
		cnt2 += 1
mu1 /= cnt1
mu2 /= cnt2

sigma1 = np.zeros((106,106))
sigma2 = np.zeros((106,106))
for i in range(train_data_size):
	if Y[i] == 1:
		sigma1 += np.dot(np.transpose([X[i] - mu1]), [(X[i] - mu1)])
	else:
		sigma2 += np.dot(np.transpose([X[i] - mu2]), [(X[i] - mu2)])
sigma1 /= cnt1
sigma2 /= cnt2
shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
N1 = cnt1
N2 = cnt2

inverse_sigma = np.linalg.pinv(shared_sigma)

W = np.dot(np.transpose(mu1-mu2),inverse_sigma)
b = -1.0*np.dot(np.dot(np.transpose(mu1),inverse_sigma),mu1)
b += 1.0*np.dot(np.dot(np.transpose(mu2),inverse_sigma),mu2)
b /= 2
b += np.log(N1/N2)

# save model
np.save("./model/model_generative_W.npy",W)
np.save("./model/model_generative_b.npy",b)
"""
#----------#
#--Output--#
#----------#

W = np.load("./model/model_generative_W.npy")
b = np.load("./model/model_generative_b.npy")

test_X = []
n_row = 0
text = open(test_url ,"r")
row = csv.reader(text , delimiter= ",")
for r in row:
	if n_row == 0:
		pass
	else:
		tmp = []
		for i in range(106):
			tmp.append(float(r[i]))
		test_X.append(tmp)
	n_row = n_row+1
text.close()
test_X = np.array(test_X)

ans = []
for i in range(len(test_X)):
	ans.append([str(i+1)])
	a = sigmoid(np.dot(W,test_X[i])+b)
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
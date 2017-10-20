import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

x = [1, 2, 3 ,7,8,9]
x = np.array(x)
# normalization
x = (x - x.min(0)) / (x.max(0) - x.min(0))
print(x)

y = sigmoid(x)

print(y)

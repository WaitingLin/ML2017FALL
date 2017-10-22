import numpy as np
import csv

x = [1,2,3,4]
w = [7,8,9]
tmp = []

x = np.array(x)
w = np.array(w)
print(x,w)

tmp = x
x = w
w = tmp

print(x,w)
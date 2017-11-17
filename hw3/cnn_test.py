import pandas as pd
import numpy as np
import csv
from keras.models import load_model
import sys

dataset_path = sys.argv[1]
output_path = sys.argv[2]

# --- Read Data --- #
data_test = pd.read_csv(dataset_path)
feature_test = data_test.feature.tolist()

X_test = []
for i in range(len(feature_test)):
	tmp = feature_test[i].split(' ')
	X_test.append(tmp)
X_test = np.array(X_test)

X_test = X_test.reshape(X_test.shape[0],48,48,1).astype('float32')
X_test_reverse = X_test[:,:,::-1,:]

# --- Preprossing --- #
X_test_norm = X_test / 255.
X_test_norm_reverse = X_test_reverse / 255

#print(X_test_norm.shape)
#print(X_test_norm_reverse.shape)

# --- Restore model --- #
model = load_model('best_model.h5')

# --- Prediction --- #
prediction1 = model.predict(X_test_norm)
prediction2 = model.predict(X_test_norm_reverse)

pre = []
for i in range(len(prediction1)):
	pre.append([str(i)])
	if np.amax(prediction1[i]) > np.amax(prediction2[i]):
		pre[i].append(np.argmax(prediction1[i]))
	else:
		pre[i].append(np.argmax(prediction2[i]))

f= open(output_path, 'w+')
s = csv.writer(f,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(pre)):
	s.writerow(pre[i])
f.close()
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.utils import np_utils  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import csv

np.random.seed(87)
# Read Data
data_train = pd.read_csv('./train.csv')
data_test = pd.read_csv('./test.csv')
label_train = data_train.label.tolist()
feature_train = data_train.feature.tolist()
feature_test = data_test.feature.tolist()

X_train = []
X_test = []
for i in range(len(feature_train)):
	tmp = feature_train[i].split(' ')
	X_train.append(tmp)
for i in range(len(feature_test)):
	tmp = feature_test[i].split(' ')
	X_test.append(tmp)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(label_train)

X_train = X_train.reshape(X_train.shape[0],48,48,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0],48,48,1).astype('float32')

# ---Preprossing--- #
X_train_norm = X_train / 255.
X_test_norm = X_test / 255.
y_train = np_utils.to_categorical(y_train)

# ---Construct model--- #
# Conv layer
model = Sequential()  
# Create CN layer 1  
model.add(Conv2D(filters=64,  
				 kernel_size=(3,3),  
				 padding='same',  
				 input_shape=(48,48,1),  
				 activation='relu'))  
 
model.add(MaxPooling2D(pool_size=(2,2)))  

# Create CN layer 2  
model.add(Conv2D(filters=32,  
				 kernel_size=(3,3),  
				 padding='same',  
				 input_shape=(48,48,1),  
				 activation='relu')) 

# Create CN layer 3  
model.add(Conv2D(filters=64,  
				 kernel_size=(3,3),  
				 padding='same',  
				 input_shape=(48,48,1),  
				 activation='relu'))  
 
model.add(MaxPooling2D(pool_size=(2,2)))  

# Create CN layer 4 
model.add(Conv2D(filters=32,  
				 kernel_size=(3,3),  
				 padding='same',  
				 input_shape=(48,48,1),  
				 activation='relu'))  

# Create CN layer 5  
model.add(Conv2D(filters=64,  
				 kernel_size=(3,3),  
				 padding='same',  
				 input_shape=(48,48,1),  
				 activation='relu'))  
 
model.add(MaxPooling2D(pool_size=(2,2)))  

# Create CN layer 6  
model.add(Conv2D(filters=128,  
				 kernel_size=(2,2),  
				 padding='same',  
				 input_shape=(48,48,1),  
				 activation='relu'))  

model.add(MaxPooling2D(pool_size=(2,2)))  

  
# Add Dropout layer  
model.add(Dropout(0.5))  


# Fully connected
model.add(Flatten())  
model.add(Dense(128, activation='relu'))  
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax')) 

#model.summary()  

# ---Training--- #
print('Training ------------')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
train_history = model.fit(x=X_train_norm,  
						  y=y_train, validation_split=0.2,  
						  epochs=30, batch_size=256, verbose=1)  
# ---Output--- #
prediction = model.predict_classes(X_test_norm)

pre = []
for i in range(len(prediction)):
    pre.append([str(i)])
    pre[i].append(prediction[i])

f= open('prediction.csv', 'w+')
s = csv.writer(f,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(pre)):
    s.writerow(pre[i])
f.close()
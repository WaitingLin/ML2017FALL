import numpy as np
import pandas as pd
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pkl
from keras.models import Sequential
from keras.utils import np_utils  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import load_model
import sys
np.random.seed(87)

train_url = sys.argv[1]

#Read Data
data_train = pd.read_csv(train_url)
#data_test = pd.read_csv('./test.csv')
label_train = data_train.label.tolist()
feature_train = data_train.feature.tolist()
#feature_test = data_test.feature.tolist()

# train
feature_train = np.array(feature_train)
f = open('X_train.pkl', 'wb')
pkl.dump(feature_train, f)
f.close()
X_train = np.zeros([len(feature_train),48,48])
for i in range(len(feature_train)):
	tmp = np.fromstring(feature_train[i], dtype=float, sep=' ') 
	tmp = tmp.reshape((48,48))
	X_train[i] = tmp
y_train = np.array(label_train)

# test
"""
X_test = []
for i in range(len(feature_test)):
	tmp = feature_test[i].split(' ')
	X_test.append(tmp)
X_test = np.array(X_test)
"""

X_train = X_train.reshape(X_train.shape[0],48,48,1).astype('float32')
#X_test = X_test.reshape(X_test.shape[0],48,48,1).astype('float32')
#X_test_reverse = X_test[:,:,::-1,:]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

f = open('X_val.pkl', 'wb')
pkl.dump(X_val, f)
f.close()
f = open('y_val.pkl', 'wb')
pkl.dump(y_val, f)
f.close()


# ---Preprossing--- #
#X_train_norm = X_train / 255.
#X_test_norm = X_test / 255.
#X_test_norm_reverse = X_test_reverse / 255

datagen = ImageDataGenerator(rescale=1./255, rotation_range=25, width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True)
valgen = ImageDataGenerator(rescale=1./255)
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

# ---Construct model--- #
# Conv layer
model = Sequential()
  
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(48,48,1), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=48, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=96, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


# Fully connected
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.3))
model.add(Dense(48, activation='relu'))  
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax')) 

model.summary()  

# ---Training--- #
adam = Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
early_stopping = EarlyStopping(monitor='val_loss', patience=64)
#train_history = model.fit(x=X_train_norm, y=y_train, validation_split=0.2, epochs=9487, batch_size=512, callbacks=[early_stopping], verbose=1)  
train_history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=256), 
		steps_per_epoch=len(X_train) / 64,epochs=400, 
		#callbacks=[early_stopping], 
		validation_data=valgen.flow(X_val, y_val), validation_steps=len(X_val) / 64, verbose=1)

# ---Output--- #
# plot 
plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('# of epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('cnn_accuracy.png')
plt.clf()
# save model 
model.save('model_best.h5')

"""
# prediction
#prediction = model.predict_classes(X_test_norm)
prediction1 = model.predict(X_test_norm)
prediction2 = model.predict(X_test_norm_reverse)

pre = []
for i in range(len(prediction1)):
	pre.append([str(i)])
	if np.amax(prediction1[i]) > np.amax(prediction2[i]):
		pre[i].append(np.argmax(prediction1[i]))
	else:
		pre[i].append(np.argmax(prediction2[i]))

f= open('prediction.csv', 'w+')
s = csv.writer(f,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(pre)):
	s.writerow(pre[i])
f.close()
"""

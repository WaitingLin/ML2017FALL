import os
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

# ---Parameter--- #
training_file = "training_label.txt"
maxlen = 64
batch_size = 256
epochs = 10
embedding_dim = 64
nb_words = 15000

# ---Read data--- #
x_train = []
y_train = []
with open(training_file, 'r') as f:
    for line in f:
        lines = line.strip().split(' +++$+++ ')
        y_train.append(int(lines[0]))
        x_train.append(lines[1])

# ---Tokenizer--- #
tokenizer = Tokenizer(num_words=nb_words, filters="\n")
tokenizer.fit_on_texts(x_train)
#print(tokenizer.word_index)
x_train = tokenizer.texts_to_matrix(x_train, mode='count')

# ---Create model--- #
model = Sequential()
model.add(Dense(2, input_dim=nb_words))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='max')
save_path = os.path.join('bow_model.h5')
checkpoint = ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True, 
                             save_weights_only=True, monitor='val_acc', mode='max')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_split=0.1, callbacks=[checkpoint, earlystopping])


strr = ["today today is a good day, but it is hot",
        "today is hot, but it is a good day"]
tokenizer.fit_on_texts(strr)
mystr = tokenizer.texts_to_matrix(strr, mode='count')
print(len(mystr))
prediction = model.predict(mystr, verbose=1)
print("prediction:")
print(strr[0],":", prediction[0])
print(strr[1],":", prediction[1])

import numpy as np
import sys, os, csv, collections
import pickle as pk
import keras
from tqdm import tqdm
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Bidirectional

# ---Parameter--- #
testing_data = "testing_data.txt"
maxlen = 128
embedding_dim = 256
# ---Read data--- #
x_test = []
n_row = 0
with open(testing_data, 'r') as f:
    for line in f:
        if n_row != 0:
            lines = line.split(',',1)
            x_test.append(lines[1])
        else:
            pass
        n_row += 1  

# ---Tokenizer--- #
with open("token.pk", "rb") as f:
    tokenizer  = pk.load(f)
x_test = tokenizer.texts_to_sequences(x_test)
nb_words = len(tokenizer.word_index)

# ---Pad Sequences--- #
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# ---Create model--- #
model = Sequential()
model.add(Embedding(nb_words, embedding_dim, embeddings_initializer=keras.initializers.random_normal(stddev=1.0)))
model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.4, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
model.load_weights('model.h5')

# ---Predict--- #
prediction = model.predict(x_test, batch_size=512, verbose=1)
#print(len(prediction))
result = [int(prediction[i]>=0.5) for i in range(len(prediction))]

# ---Output--- #
with open("prediction.csv", "w+") as f:
    w = csv.writer(f, delimiter=',', lineterminator='\n')
    w.writerow(["id","label"])
    for i in range(len(result)):
        w.writerow([i,result[i]])

strr = ["today today is a good day, but it is hot",
        "today is hot, but it is a good day"]
tokenizer.fit_on_texts(strr)
mystr = tokenizer.texts_to_matrix(strr, mode='count')
print(len(mystr))
pre = model.predict(mystr, batch_size=512, verbose=1)
print("prediction:")
print(strr[0],":", pre[0])
print(strr[1],":", pre[1])

import numpy as np
import pickle as pk
import argparse, sys, os
import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from stemming.porter2 import stem
#from tqdm import tqdm

parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('--filter', default='n', choices=['all','n'])
args = parser.parse_args()

# ---Parameter--- #
training_file = "training_label.txt"
maxlen = 128
batch_size = 512 #128
epochs = 10
embedding_dim = 256

# ---Read data--- #
x_train = []
y_train = []
with open(training_file, 'r') as f:
    for line in f:
        lines = line.strip().split(' +++$+++ ')
        y_train.append(int(lines[0]))
        #lines[1] = stem(lines[1])
        x_train.append(lines[1])

# ---Tokenizer--- #
if args.filter == 'n':
    tokenizer = Tokenizer(filters="\n")
    #tokenizer = Tokenizer(num_words=nb_words)
else:
    tokenizer = Tokenizer()
    #tokenizer = Tokenizer(num_words=nb_words)
tokenizer.fit_on_texts(x_train)
#print(tokenizer.word_index)
pk.dump(tokenizer, open("token.pk", 'wb'))
x_train = tokenizer.texts_to_sequences(x_train)
nb_words = len(tokenizer.word_index)

# ---Pad sequences--- #   
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

# ---Model--- #
model = Sequential()
model.add(Embedding(nb_words, embedding_dim, embeddings_initializer=keras.initializers.random_normal(stddev=1.0)))
model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.4, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# ---Training--- #
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
save_path = os.path.join('model.h5')
earlystopping = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True, 
                             save_weights_only=True, monitor='val_acc', mode='max')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_split=0.1, callbacks=[checkpoint, earlystopping])


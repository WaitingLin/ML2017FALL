import keras
import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Flatten, Dot, Add, Concatenate, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping

def read_data(path):
    df = pd.read_csv(path)
    df = df.values
    users = []
    movies = []

    if path == './data/train.csv':
        rating = []
        label = []
        np.random.shuffle(df)
        for i in range(len(df)):
            users.append(int(df[i][1]))
            movies.append(int(df[i][2]))
            rating.append(float(df[i][3]))        
        users, movies, rating = np.array(users), np.array(movies), np.array(rating)
        return users, movies, rating
    
    for i in range(len(df)):
        users.append(int(df[i][1]))
        movies.append(int(df[i][2]))
    users, movies = np.array(users), np.array(movies)
    return users, movies

def mf(n_users, n_movies, latent_dim = 32):
    user_input = Input(shape=[1])
    movie_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    movie_vec = Embedding(n_movies, latent_dim, embeddings_initializer='random_normal')(movie_input)
    movie_vec = Flatten()(movie_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    movie_bias = Embedding(n_movies, 1, embeddings_initializer='zeros')(movie_input)
    movie_bias = Flatten()(movie_bias)
    r_hat = Dot(axes=1)([user_vec, movie_vec])
    r_hat = Add()([r_hat, user_bias, movie_bias])
    model = Model([user_input, movie_input], r_hat)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def dnn(n_users, n_movies, latent_dim = 32):
    user_input = Input(shape=[1])
    movie_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    movie_vec = Embedding(n_movies, latent_dim, embeddings_initializer='random_normal')(movie_input)
    movie_vec = Flatten()(movie_vec)
    merge_vec = Concatenate()([user_vec, movie_vec])
    hidden = Dense(256, activation='relu')(merge_vec)
    hidden = Dropout(0.7)(hidden)
    output = Dense(1)(hidden)
    model = Model([user_input, movie_input], output)
    model.compile(loss='mse', optimizer='adam')
    model.summary() 
    return model

def main(isNormalize=False, isDNN=False):
    users, movies, rating = read_data('./data/train.csv')
    
    # normalize
    if isNormalize:
        mean = np.mean(rating)
        std = np.std(rating)
        rating = (rating-mean) / std
    
    # mf or dnn
    if isDNN:
        model = dnn(6040, 3952, 512)
    else:
        model = mf(6040, 3952, 32)
    patience = 1
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
    model.fit([users, movies], rating, epochs=87, batch_size=512, verbose=1,
              validation_split=0.1, callbacks=[early_stopping])
    model.save('./model.h5') 
    print('save model as modle.h5')

    users_test , movies_test = read_data('./data/test.csv')
    prediction = model.predict([users_test, movies_test], batch_size=512, verbose=1)
    if isNormalize:
        prediction = prediction * std + mean
    prediction = np.clip(prediction, 1, 5)
    index = np.arange(len(prediction))+1
    prediction = prediction.reshape(-1)
    df = pd.DataFrame.from_items([('TestDataID',index.tolist()),('Rating',prediction.tolist())])
    df.to_csv('./predict_withbias.csv',index=False)
    print('save prediction as predict.csv')

if __name__ == '__main__':
    main(isNormalize=False, isDNN=False)

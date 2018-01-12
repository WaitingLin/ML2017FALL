import numpy as np
import pandas as pd

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.cluster import KMeans
from keras.callbacks import EarlyStopping

def autoencoder(path, encoding_dim):
    X = np.load(path) / 255.
    print('X.shape: ', X.shape)	
    
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='tanh')(decoded)
    
    autoencoder = Model(input_img, decoded)
	
    # encoder
    encoder = Model(input_img, encoded)


    autoencoder.compile(optimizer='adam', loss='mse')
    
    autoencoder.fit(X, X,epochs=200,batch_size=256,shuffle=True)#,validation_split=0.1, callbacks=[early_stopping])
    encoder.save('./model.h5')

    Xm = encoder.predict(X)
    kmeans = KMeans(n_clusters=2).fit(Xm)
    labels = kmeans.labels_
    df = pd.read_csv('./test_case.csv')
    df = df.values
    
    ID = []
    Image1_ID = []
    Image2_ID = []
    for i in range(len(df)):
        ID.append(int(df[i][0]))
        Image1_ID.append(int(df[i][1]))
        Image2_ID.append(int(df[i][2]))
    ID, Image1_ID, Image2_ID = np.array(ID), np.array(Image1_ID), np.array(Image2_ID)

    ans = []
    print(labels[Image1_ID[0:30]])
    print(labels[Image2_ID[0:30]])
    for i in range(len(ID)):
        if labels[Image1_ID[i]]==labels[Image2_ID[i]]:
            ans.append(1)
        else:
            ans.append(0)
    df = pd.DataFrame.from_items([('ID',ID),('Ans',ans)])
    df.to_csv('./prediction.csv',index=False)
    print('ans:',ans[0:30])

if __name__ == '__main__':
    autoencoder('./image.npy', 32)

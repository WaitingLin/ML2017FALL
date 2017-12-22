import sys
import pandas as pd
import numpy as np
from keras.models import load_model

test_data_path = sys.argv[1]
predict_file_path = sys.argv[2]
load_model_path = sys.argv[3]

def read_data(path):
    df = pd.read_csv(path)
    df = df.values
    users = []
    movies = []
    for i in range(len(df)):
        users.append(int(df[i][1]))
        movies.append(int(df[i][2]))
    users, movies = np.array(users), np.array(movies)
    return users, movies

def main():
    users_test, movies_test = read_data(test_data_path)
    model = load_model(load_model_path)
    prediction = model.predict([users_test, movies_test], batch_size=512, verbose=1)
    index = np.arange(len(prediction))+1
    prediction = prediction.reshape(-1)
    df = pd.DataFrame.from_items([('TestDataID',index.tolist()),('Rating',prediction.tolist())])
    df.to_csv(predict_file_path, index=False)

if __name__ == '__main__':
    main()

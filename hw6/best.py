import numpy as np
import pandas as pd
import sys
from keras.models import load_model
from sklearn.cluster import KMeans

def main(image_path, test_case_path, prediction_file):
    encoder = load_model('best.h5')
    X = np.load(image_path) / 255
    print('X.shape: ', X.shape) 

    Xm = encoder.predict(X)
    kmeans = KMeans(n_clusters=2).fit(Xm)
    labels = kmeans.labels_
    df = pd.read_csv(test_case_path)
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
    for i in range(len(ID)):
        if labels[Image1_ID[i]]==labels[Image2_ID[i]]:
            ans.append(1)
        else:
            ans.append(0)
    df = pd.DataFrame.from_items([('ID',ID),('Ans',ans)])
    df.to_csv(prediction_file, index=False)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

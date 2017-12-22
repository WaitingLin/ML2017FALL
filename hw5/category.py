import numpy as np

from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def read_data(path='./data/movies.csv'):
    f = open(path, 'r', encoding='latin-1')
    read_data = f.readlines()
    for i in range(len(read_data)):
        read_data[i] = read_data[i].split('::')
        read_data[i][2] = read_data[i][2][:-1]
        read_data[i][2] = read_data[i][2].split('|')[0]

    category = {'Comedy': 1, 'Fantasy': 1, 'Romance': 1, 'Drama': 1, 'Musical': 1, 
                'Thriller': 3, 'Horror': 3, 'Crime': 3, "Mystery": 3, 'Action': 2,
                'Sci-Fi': 1, 'Documentary': 2, 'War': 3, 'Adventure': 3, 'Animation': 1, "Children's": 1,
                'Western': 2, 'Film-Noir': 3}
    dictionary = {}
    for i in range(1,len(read_data)):
        dictionary[read_data[i][0]] = category[read_data[i][2]] 
    f.close()
    return dictionary

def draw(x, y):
    y = np.array(y)
    x = np.array(x, dtype=np.float64)

    vis_data = TSNE(n_components=2).fit_transform(x)

    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=y)
    plt.colorbar(sc)
    plt.show()

def main():
    dictionary = read_data('./data/movies.csv')
    model = load_model('model.h5')
    movie_emb = np.array(model.layers[3].get_weights()).squeeze()
    label=[]
    for i in range(len(movie_emb)):
        if str(i+1) in dictionary:
            label.append(dictionary[str(i+1)])
        else:
            label.append(0)
    draw(movie_emb, label)
    
if __name__ == '__main__':
    main()

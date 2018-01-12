import matplotlib
#matplotlib.use('Agg')

from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from adjustText import adjust_text

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

font_path = 'H-MingLan-R.ttf'
font_prop = matplotlib.font_manager.FontProperties(fname=font_path)


def plot(Xs, Ys, Texts):
    plt.plot(Xs, Ys, 'o')
    texts = [plt.text(X,Y,Text, fontproperties=font_prop) for X, Y, Text in zip(Xs, Ys, Texts)]
    plt.title(str(adjust_text(texts, Xs, Ys, arrowprops=dict(arrowstyle='->', color='red'))))
    plt.show()
    
def main():
    K = 6000

    model = Word2Vec.load('250.model.bin')
        
    words = open('seg.txt')
    words = words.readline()
    words = words.split(' ')
    
    dictionary = {}
    for i in range(len(words)):
        if words[i] not in dictionary:
            dictionary[words[i]] = 1
        else:
            dictionary[words[i]] = dictionary[words[i]] + 1
    word = []
    for key in dictionary:
        if dictionary[key] > K:
            word.append(key)                
    vector = []
    
    for i in word:
        vector.append(model.wv[i])
    vector = np.array(vector)
    print(vector.shape)

    embedded = TSNE(n_components=2).fit_transform(vector)
   
    print(embedded)
    x = []
    y = []
    for value in embedded:
        x.append(value[0])
        y.append(value[1])

    print(x)
    print(y)
    plot(x, y, word)

if __name__ == '__main__':
    main()

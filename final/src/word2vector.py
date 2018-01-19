from gensim.models import word2vec
import logging
import sys, os

size = int(sys.argv[1])
window = int(sys.argv[2])

def main():
    if not os.path.exists('./model'):
        os.makedirs('./model')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus('./segment/seg_all.txt')
    model = word2vec.Word2Vec(sentences, size=size, window=window, sg=1)
    model.save('./model/'+sys.argv[1]+'_'+sys.argv[2]+'.model.bin')
if __name__ == '__main__':
    main()

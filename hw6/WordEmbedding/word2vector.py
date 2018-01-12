from gensim.models import word2vec


def main():
    sentences = word2vec.Text8Corpus("seg.txt")
    model = word2vec.Word2Vec(sentences, size=250)
    
    model.save("250.model.bin")


if __name__ == "__main__":
    main()

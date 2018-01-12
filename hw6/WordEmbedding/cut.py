import jieba
import numpy as np

def main():
    jieba.set_dictionary('dict.txt.big')

    output = open('seg.txt','w')

    with open('all_sents.txt', 'r') as content:
        for line in content:
            line = line[:-1]
            words = jieba.cut(line, cut_all=False)
            for word in words:
                output.write(word +' ')
    output.close()


if __name__ == '__main__':
    main()

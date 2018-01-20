import jieba, logging, os, sys
import numpy as np

def main():
    if not os.path.exists('./segment'):
        os.makedirs('./segment')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    jieba.set_dictionary('data/dict.txt.big')
    stopwordset = set()
    with open('jieba_dict/stopwords.txt','r',encoding='utf-8') as sw:
        for line in sw:
            stopwordset.add(line.strip('\n'))
    dirs = sys.argv[1]
    data_path = [dirs+'/1_train.txt',dirs+'/2_train.txt',dirs+'/3_train.txt',
                 dirs+'/4_train.txt',dirs+'/5_train.txt']
    data = []
    output = open('./segment/seg_all.txt', 'w')
    print('jieba cut')
    for i in range(len(data_path)):
        with open(data_path[i], 'r') as f:
            for line in f:
                lines = line[:-1]
                data.append(lines) 
                
    all_text = ''.join(data)
    words = jieba.cut(all_text, cut_all=False)

    n_row = 0
    for word in words:
        output.write(word+' ')
        n_row += 1
        if n_row % 10000 == 0:
            logging.info("%d rows finished" %n_row)
    output.close()
    
if __name__ == '__main__':
    main()

from gensim.models import Word2Vec
import jieba
import numpy as np
import sys, os

if not os.path.exists('./result'):
    os.makedirs('./result')

# Embedding dimension
EBD_DIM = int(sys.argv[1])
model = "./model/"+sys.argv[1]+"_"+sys.argv[2]+".model.bin"
save_path = "./prediction/"+sys.argv[1]+"_"+sys.argv[2]+".csv"

jieba.set_dictionary('./data/dict.txt.big')

ftest = open("./data/testing_data.csv", "r")
ftest.readline()

Question = []
Options = []

while True:
    line = ftest.readline()

    if (not line) or line == '':
        break

    lines = line[:-1].split(',')

    # Question
    questions = lines[1].split('\t')
    question = []
    for q in questions:
        question.append(q[2:])
    question = ' '.join(question)

    # Options
    options = []
    temp = lines[2].split('\t')
    for tmp in temp:
        tmp = tmp[2:]
        options.append(tmp)

    Question.append(question)
    Options.append(options)

# Storing jieba result
jieba_Question = []
jieba_Options = []

model = Word2Vec.load(model)

for Q in Question:
    words = jieba.cut(Q, cut_all = False)
    jieba_Question.append(words)

for Opt in Options:
    temp = []
    for item in Opt:
        words = jieba.cut(item, cut_all = False)
        temp.append(words)
    jieba_Options.append(temp)

# Produce Sentence Vector (Just add them all)
sentence_Question = []
sentence_Options = []

for Q in jieba_Question:
    result = np.zeros(EBD_DIM)
    count = 0
    for word in Q:
        if word in model.wv:
            result += model.wv[word]
        count += 1
    sentence_Question.append(result / count)

for Opt in jieba_Options:
    tmp = []
    for item in Opt:
        result = np.zeros(EBD_DIM)
        count = 0
        for word in item:
            if word in model.wv:
                result += model.wv[word]
            count += 1
        tmp.append(result / count)
    sentence_Options.append(tmp)

fout = open(save_path, "w")
fout.write("id,ans\n")

for i in range(0, len(sentence_Question)):
    q = sentence_Question[i]
    distance_list = []
    print(Question[i])
    print(Options[i])
    for opt in sentence_Options[i]:
        dis = np.dot(q, opt) / (np.linalg.norm(q) * np.linalg.norm(opt))
        distance_list.append(dis)
    answer = distance_list.index(max(distance_list))
    fout.write(str(i + 1) + "," + str(answer) + "\n")
    print("Answer is", answer)

fout.close()

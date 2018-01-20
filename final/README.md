# Machine Learning Final Project
## 版本與套件
* Python 3.5.2
* Jieba斷詞
* Gensim做word to vector

## Run best model 
* python3 answer.py \<size> \<window> \<testing_data path>
      
      python3 answer.py 100 25 ./data/testing_data.csv

## Training
[1] 執行segment_all.py，將所有的training data斷詞，結果會存到./segment/seg_all.txt中。
* python3 segment_all.py \<data path>
* data資料夾中需要包含五個training file: 1_train.txt, 2_train.txt, 3_train.txt, 4_train.txt, 5_train.txt

      python3 segment_all.py ./data

[2] 執行word2vector.py，將seg_all.txt當做Word2Vec的sentences，model會存到./model中
* python3 word2vector.py \<size> \<window>
* 兩個參數: size, window

      python3 word2vector.py 100 25

## Predict
執行answer.py，將題目轉成
* python3 answer.py \<size> \<window> \<testing_data path>
* 兩個參數: size, window

      python3 answer.py 100 25 ./data/testing_data.csv

# Machine Learning Final Project
## Run best model 
    python3 answer.py 100 25

## Training的部分
[1] 執行segment_all.py，將所有的training data斷詞，結果會存到./segment/seg_all.txt中。
* python3 segment_all.py \<data path>
* data資料夾中需要包含五個training file: 1_train.txt, 2_train.txt, 3_train.txt, 4_train.txt, 5_train.txt

      python3 segment_all.py ./data

[2] 執行word2vector.py，
* 兩個參數: size, window

      python3 word2vector.py 100 25

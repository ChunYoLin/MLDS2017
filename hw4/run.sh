# !/bin/bash
wget https://www.cmlab.csie.ntu.edu.tw/~kainbd/HW4_Model.tar.gz
tar zxvf HW4_Model.tar.gz
time python main.py --train=False --input_file=$2 --output_file=$3

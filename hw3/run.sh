# !/bin/bash
wget https://www.csie.ntu.edu.tw/~r04922108/MLDS/hw3_model/wgan_new.tar.gz
tar zxvf wgan_new.tar.gz
python generator.py $1

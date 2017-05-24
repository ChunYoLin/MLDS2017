# !/bin/bash
wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt -P skipthoughts/model/
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy -P skipthoughts/model/
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy -P skipthoughts/model/
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz -P skipthoughts/model/
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl -P skipthoughts/model/
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz -P skipthoughts/model/
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl -P skipthoughts/model/
wget https://www.csie.ntu.edu.tw/~r04922108/MLDS/hw3_model/wgan_new.tar.gz
tar zxvf wgan_new.tar.gz
python generate.py $1

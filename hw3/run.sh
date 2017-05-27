# !/bin/bash
# wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt -P skipthoughts/models/
# wget http://www.cs.toronto.edu/~rkiros/models/utable.npy -P skipthoughts/models/
# wget http://www.cs.toronto.edu/~rkiros/models/btable.npy -P skipthoughts/models/
# wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz -P skipthoughts/models/
# wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl -P skipthoughts/models/
# wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz -P skipthoughts/models/
# wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl -P skipthoughts/models/
wget https://www.csie.ntu.edu.tw/~r04922108/MLDS/hw3_model/wgan.tar.gz
tar zxvf wgan.tar.gz
python generate.py $1

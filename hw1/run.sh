wget https://www.csie.ntu.edu.tw/~r04922108/MLDS/hw1_model/MODEL_500P_50000V.tar.gz
tar zxvf MODEL_500P_50000V.tar.gz
python lstm_model.py --train=False --posts=500 --vocab_size=50000 --get_input=pk --lstm_model=MODEL_500P_50000V/basic_lstm/ --test_path=$1 --out=$2

wget https://www.cmlab.csie.ntu.edu.tw/~kainbd/HW2_Model.tar.gz
tar zxvf HW2_Model.tar.gz
python S2VT_Attent.py \
 --train=False --save_path="HW2_Model/v3.ckpt" \
 --train_label="HW2_Model/training_label.json" \
 --id_file=$1 --feat_path=$2

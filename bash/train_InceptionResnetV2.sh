#########################################################################
# File Name: train_InceptionResnetV2.sh
# Author: Wenbo Tang
# mail: tangwenbo1995@163.com
# Created Time: 2018年03月25日 星期日 02时07分24秒
#########################################################################
#!/bin/bash
source activate py35
python ../code35/train/train_InceptionResnetV2.py --mission_id=1 --BATCH_SIZE=16 --EPOCH=90 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='30,55,80'
python ../code35/train/train_InceptionResnetV2.py --mission_id=2 --BATCH_SIZE=16 --EPOCH=90 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='30,55,80'
python ../code35/train/train_InceptionResnetV2.py --mission_id=3 --BATCH_SIZE=16 --EPOCH=90 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='30,55,80'
python ../code35/train/train_InceptionResnetV2.py --mission_id=4 --BATCH_SIZE=16 --EPOCH=90 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='30,55,80'
python ../code35/train/train_InceptionResnetV2.py --mission_id=5 --BATCH_SIZE=16 --EPOCH=90 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='30,55,80'
python ../code35/train/train_InceptionResnetV2.py --mission_id=6 --BATCH_SIZE=16 --EPOCH=90 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='30,55,80'
python ../code35/train/train_InceptionResnetV2.py --mission_id=7 --BATCH_SIZE=16 --EPOCH=90 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='30,55,80'
python ../code35/train/train_InceptionResnetV2.py --mission_id=8 --BATCH_SIZE=16 --EPOCH=90 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='30,55,80'

#########################################################################
# File Name: train_googlev3.sh
# Author: Wenbo Tang
# mail: tangwenbo1995@163.com
# Created Time: 2018年03月23日 星期五 18时39分01秒
#########################################################################
#!/bin/bash
#python_file = /media/tang/code/tianchi/code/train/train_model.py 
python ../code/train/train_model.py --mission_id=1 --BATCH_SIZE=46 --EPOCH=80 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='25, 50, 65'
python ../code/train/train_model.py --mission_id=2 --BATCH_SIZE=46 --EPOCH=80 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='25, 50, 65'
python ../code/train/train_model.py --mission_id=3 --BATCH_SIZE=46 --EPOCH=80 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='25, 50, 65'
python ../code/train/train_model.py --mission_id=4 --BATCH_SIZE=46 --EPOCH=80 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='25, 50, 65'
python ../code/train/train_model.py --mission_id=5 --BATCH_SIZE=46 --EPOCH=80 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='25, 50, 65'
python ../code/train/train_model.py --mission_id=6 --BATCH_SIZE=46 --EPOCH=80 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='25, 50, 65'
python ../code/train/train_model.py --mission_id=7 --BATCH_SIZE=46 --EPOCH=80 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='25, 50, 65'
python ../code/train/train_model.py --mission_id=8 --BATCH_SIZE=46 --EPOCH=80 --LR=0.01 --LR_DECAY=0.1 --DECAY_EPOCH='25, 50, 65'

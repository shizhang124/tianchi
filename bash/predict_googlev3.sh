#########################################################################
# File Name: predict_googlev3.sh
# Author: Wenbo Tang
# mail: tangwenbo1995@163.com
# Created Time: 2018年03月24日 星期六 14时38分43秒
#########################################################################
#!/bin/bash
python ../code/train/predict_model.py --mission_id=1 --BATCH_SIZE=46 
python ../code/train/predict_model.py --mission_id=2 --BATCH_SIZE=46 
python ../code/train/predict_model.py --mission_id=3 --BATCH_SIZE=46 
python ../code/train/predict_model.py --mission_id=4 --BATCH_SIZE=46 
python ../code/train/predict_model.py --mission_id=5 --BATCH_SIZE=46 
python ../code/train/predict_model.py --mission_id=6 --BATCH_SIZE=46 
python ../code/train/predict_model.py --mission_id=7 --BATCH_SIZE=46 
python ../code/train/predict_model.py --mission_id=8 --BATCH_SIZE=46 

bash merge_all_result.sh
echo "have merge all results"

#########################################################################
# File Name: merge_all_result.sh
# Author: Wenbo Tang
# mail: tangwenbo1995@163.com
# Created Time: 2018年03月24日 星期六 15时52分42秒
#########################################################################
#!/bin/bash
#Folder_A="../predicts"    
#for file_a in ${Folder_A}/*  
#do    
#    temp_file=`basename $file_a`    
#    echo $temp_file    
#    cat $tmp_file>>"../predicts/all_predicts.csv"
#done    
rm -f ../predicts/all_result.csv
cat ../predicts/*.csv >> ../predicts/all_result.csv

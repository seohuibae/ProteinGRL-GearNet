#!/bin/bash
DirName=$(python run_parallel_mkdir_downstream.py) 
echo $DirName
wait 
python downstream.py --run_dir $DirName --dataset FC-Fold --model gcn --batch_size 2 --seed 2020 --gpus 0 --load_dir results/pretrain/2022-09-30_03-45-52 --start_from -1 --type angle     
wait 
python run_parallel_read.py --run_dir $DirName 

    

# python downstream.py --run_dir $DirName --dataset FC-Fold --model gcn --batch_size 2 --seed 2020 --gpus 0  &    
# python downstream.py --run_dir $DirName --dataset FC-Fold --model gcn --batch_size 2 --seed 2021 --gpus 1  &    
# python downstream.py --run_dir $DirName --dataset FC-Fold --model gcn --batch_size 2 --seed 2022 --gpus 2  & 
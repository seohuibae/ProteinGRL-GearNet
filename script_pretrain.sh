#!/bin/bash
DirName=$(python run_parallel_mkdir_pretrain.py) 
echo $DirName
wait 
python pretrain.py --run_dir $DirName --model gcn --batch_size 96 --type angle --seed 2020 --gpus 0 
wait 

# python pretrain.py --run_dir $DirName --model gcn --batch_size 96 --type angle --seed 2021 --gpus 1  &    
# python pretrain.py --run_dir $DirName --model gcn --batch_size 96 --type angle --seed 2022 --gpus 2  &    
    

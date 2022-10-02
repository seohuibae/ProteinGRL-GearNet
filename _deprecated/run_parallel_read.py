import os 
import argparse 
import json 
import glob 
import numpy as np 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default='')
    args = parser.parse_args()
    assert args.run_dir != ''
    
    json_files = []
    for file in glob.glob(args.run_dir+"/*.json"):
        json_files.append(file)
        print(file)

    vaccs, taccs= [], []
    for fpath in json_files: 
        with open(fpath, 'r', encoding="UTF-8") as f: 
            json_dict = json.load(f.read())
            vaccs.append(json_dict['veval'])
            taccs.append(json_dict['teval'])
            exp_config = json_dict['exp_config']
            args_dict = json_dict['args']
    
    if len(vaccs)==0 and len(taccs)==0:
        print('empty')
        exit() 

    vacc = [np.mean(vaccs), np.std(vaccs)]
    tacc = [np.mean(taccs), np.std(taccs)]

    print("-------------------------------------")
    print(args.run_dir)
    print(exp_config)
    print(f"val: {vacc[0]*100:.2f}({vacc[1]*100:.2f})")
    print(f"test: {tacc[0]*100:.2f}({tacc[1]*100:.2f})")
    print(args_dict)

    print("-------------------------------------")

    with open(args.run_dir+'/total.txt', 'w') as f:
        f.write(exp_config+'\n')
        f.write(str(args_dict)+'\n')
        f.write(f"val: {vacc[0]*100:.2f}({vacc[1]*100:.2f})\n")
        f.write(f"test: {tacc[0]*100:.2f}({tacc[1]*100:.2f})\n")
        
    
    exit()
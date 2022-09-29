import os 
import datetime 

if __name__ =="__main__":
    ROOT_DIR = './results/pretrain/'
    dirname = os.path.join(ROOT_DIR, datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(dirname)
    print(dirname)
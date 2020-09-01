import argparse
import time
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=None)
args = parser.parse_args()

GNN_dir = './Log/' + args.dir + '/GNN/'
print(GNN_dir)
com = input(f"want to clean {GNN_dir} ? (y/n)")
if com == 'y':
    while True:
        if os.path.exists(GNN_dir):
            shutil.rmtree(GNN_dir)
        os.makedirs(GNN_dir)
        time.sleep(3600)

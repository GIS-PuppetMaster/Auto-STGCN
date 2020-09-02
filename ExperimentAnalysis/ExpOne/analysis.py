import pickle as pk
from ExperimentDataLogger import Logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
log = {}
with open("../../Log/PEMS03_experiment1_qlearning/logger.pkl", "rb") as f:
    log['qlearning'] = pk.load(f)
for name, logger in log.items():
    data_unit = np.array(logger.data_unit)
    arr = []
    for data in data_unit:
        arr.append([data[1], data[-2][-1], data[0], data[2],data[3],data[4], data[6]])
    arr = np.array(arr)
    arr = arr[np.argsort(arr[:, 1])]
    print(name)
    print(f'action:{arr[-1, 0]} reward:{arr[-1,1]}')
    print(f'action:{arr[-2, 0]} reward:{arr[-2,1]}')


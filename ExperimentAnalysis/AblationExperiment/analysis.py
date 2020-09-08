import pickle as pk
from ExperimentDataLogger import Logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

dirs = ['PEMS03_experiment2_qlearning_2_phase1', 'PEMS03_experiment2_qlearning_2_phase2',
        'PEMS03_experiment2_qlearning_2_phase3']
for dir in dirs:
    path = f"../../Log/{dir}/logger.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            logger = pk.load(f)
        data_unit = np.array(logger.data_unit)
        arr = []
        for data in data_unit:
            arr.append(data[4])
        arr = np.squeeze(np.array(arr))
        arr = arr[np.argsort(arr[:, 1])]
        print(
            f'{dir}最优test结果:loss{arr[0][0]}, MAE:{arr[0][1]}, MAPE:{arr[0][2]}, RMSE:{arr[0][3]}, Time:{arr[:, 4].mean()}')

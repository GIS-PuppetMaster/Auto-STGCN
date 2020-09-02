import pickle as pk
from ExperimentDataLogger import Logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with open("../../Log/PEMS08_experiment2_qlearning_2/logger.pkl", "rb") as f:
    logger = pk.load(f)
data_unit = np.array(logger.data_unit)
arr = []
for data in data_unit:
    arr.append(data[4])
arr = np.squeeze(np.array(arr))
arr = arr[np.argsort(arr[:, 1])]
print(f'最优test结果:loss{arr[0][0]}, MAE:{arr[0][1]}, MAPE:{arr[0][2]}, RMSE:{arr[0][3]}, Time:{arr[:,4].mean()}')

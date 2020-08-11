import pickle as pk
from ExperimentDataLogger import Logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with open("../../Log/experiment_0/logger.pkl", "rb") as f:
    logger = pk.load(f)

data_unit = logger.data_unit
MAE = []
Time = []
training_time = []
for episode in range(len(data_unit)):
    if data_unit[episode][2][-1] is not None:
        training_time.append(data_unit[episode][2][-1][-1])
    buffer = data_unit[episode][3]
    if buffer is not None and None not in buffer:
        buffer = np.array(buffer).squeeze()
        MAE.append(buffer[1])
        Time.append(buffer[4])
print(f"training_time:{training_time}\nmean_training_time:{np.array(training_time).mean()}")

tmp = np.array(MAE)
remove_index_list = [0 for _ in range(len(MAE))]
for i in range(len(MAE)):
    mae = MAE[i]
    remove_index_list[i] = 1
    tmp_arr = np.ma.masked_array(tmp, mask=remove_index_list)
    if not np.abs(mae - tmp_arr.mean()) > 3 * tmp_arr.std():
        remove_index_list[i] = 0
offset = 0
for idx, mask in enumerate(remove_index_list):
    if mask == 1:
        del MAE[idx - offset]
        del Time[idx - offset]
        offset += 1
MAE = np.array(MAE)
Time = np.array(Time)
ax = sns.lineplot(x=Time, y=MAE)
plt.show()

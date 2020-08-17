import pickle as pk
from ExperimentDataLogger import *
import pandas as pd

with open('PEMS03_experiment1_prioritized_last_/logger.pkl', 'rb') as f:
    logger = pk.load(f)
times = []
for _, _, train, eval, test, _, time in logger.data_unit:
    if eval is not None and None not in eval:
        times.append(np.array(eval)[0, -1])
time = pd.DataFrame(np.array(times))
print(time.describe())

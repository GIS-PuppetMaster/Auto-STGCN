from math import *
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
epsilon = 0.9
print(epsilon)
x=[]
y=[]
for i in range(0, 2500):
    x.append(i)
    y.append(epsilon)
    epsilon *= pow(0.9999, i/50)
    print(f'{i}:{epsilon}')
seaborn.lineplot(data=pd.DataFrame(np.array([x,y]).T,columns=["episode","epsilon"]),x='episode',y="epsilon")
plt.show()
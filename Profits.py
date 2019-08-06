import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Data/Code_Data/profits.csv').astype('float32')

buy_and_hold = np.array(data["buy_and_hold"])

trader_daily_profit = np.array(data["long and short"])
trader_total_profit = np.array(data["landsprof"])

investor_profit = np.array(data["profit:"])

plt.figure()
plt.title("Strategy 1")
plt.plot(buy_and_hold, label = "Buy and hold")
plt.plot(investor_profit, label = "Strategy 1")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")

plt.figure()
plt.title("Strategy 2")
plt.plot(buy_and_hold, label = "Buy and hold")
plt.plot(trader_daily_profit, label = "Strategy 2")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")

plt.figure()
plt.title("Strategy 2b")
plt.plot(buy_and_hold, label = "Buy and hold")
plt.plot(trader_total_profit, label = "Strategy 2")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


'''Learning Rate'''
x = np.arange(-10,10,0.1)
y = np.arange(-10,10,0.1)
X, Y = np.meshgrid(x, y)
R = (X**2*Y**2 + 16)
Z = R

theta_1 = 1
theta_2 = 1
alpha = 0.25
theta1s = [1]
theta2s = [1]

While (theta_1 >0) and (theta_2>0):
    dj_d1 = 2*theta_1*theta_2**2
    dj_d2 = 2*theta_1**2*theta_2
    theta_1,theta_2 = theta_1 - alpha*(dj_d1),theta_2 - alpha*(dj_d2)
    theta1s.append(theta_1)
    theta2s.append(theta_2)
    if (theta_1 == 0) and (theta_2 == 0):
        break

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

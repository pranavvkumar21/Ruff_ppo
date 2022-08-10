#! /usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pandas as pd
from ta.trend import EMAIndicator

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
log_file = "ppo_ruff_logfile.csv"
df = pd.read_csv(log_file)
def animate(i):
    df = pd.read_csv(log_file)
    xar = df["episode"].values
    yar = df["returns"].values
    ema10 = EMAIndicator(close=df["returns"],window=100)
    ema = ema10.ema_indicator().values
    ax1.clear()
    ax1.plot(xar,yar)
    ax1.plot(xar,ema)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
print(df.keys())

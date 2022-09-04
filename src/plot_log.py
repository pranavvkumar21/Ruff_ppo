#! /usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pandas as pd
from ta.trend import EMAIndicator

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax3 = fig.add_subplot(2,1,1)
log_file = "ppo_ruff_logfile.csv"
df = pd.read_csv(log_file)
def animate(i):
    df = pd.read_csv(log_file)
    xar = df["episode"].values
    yar_act = df["act_loss"].values
    yar_crit = df["crit_loss"].values
    yar_eps = = df["eps_reward"].values
    ema10_a = EMAIndicator(close=df["act_loss"],window=100)
    ema_a = ema10_a.ema_indicator().values
    ema10_c = EMAIndicator(close=df["crit_loss"],window=100)
    ema_c = ema10_c.ema_indicator().values
    ema10_e = EMAIndicator(close=df["eps_reward"],window=100)
    ema_e = ema10_e.ema_indicator().values
    ax1.clear()
    ax1.plot(xar,yar_act)
    ax1.plot(xar,ema_a)
    ax2.clear()
    ax2.plot(xar,yar_crit)
    ax2.plot(xar,ema_c)
    ax3.clear()
    ax3.plot(xar,yar_eps)
    ax3.plot(xar,ema_e)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
print(df.keys())

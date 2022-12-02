#! /usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pandas as pd
from ta.trend import EMAIndicator
import numpy as np
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
log_file = "../logs/ruff_reward_log.csv"
df = pd.read_csv(log_file)
maxy = float("-inf")
val = -1
def animate(i):
    df = pd.read_csv(log_file)
    #xar = df["episode"].values
    df_new = df.iloc[:,4:].values
    #print(df.iloc[-1,:].values)
    #print("-------")
    df_new = np.sum(df_new,axis=1)
    global val
    if df_new[-1]!=val:
        print(len(df))
        val = df_new[-1]
        print(np.sum(df_new[-1]))
    yar_eps = df["forward_velocity"].values
    #steps = (df["step"].values)//10
    ema10_e = EMAIndicator(close=df["forward_velocity"],window=100)
    ema_e = ema10_e.ema_indicator().values
    ax1.clear()
    ax1.plot(yar_eps)
    ax1.plot(ema_e)
    #ax1.scatter(xar,steps,c="g")
    global maxy
    if np.max(yar_eps)>maxy:
        maxy = np.max(yar_eps)
        print("new maximum reward: "+str(maxy))
        maxx = np.argmax(yar_eps)
        print("achieved at episode: "+str(maxx))
        print("-"*20)
    #print(sum(df["step"].values))
    #ax1.scatter([maxx],[maxy],c="r")
#    plt.ylim(-15,10)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
print(sum(df["step"].values))

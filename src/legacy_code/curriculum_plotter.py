#!/usr/bin/env python3

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
kc = 0.00000000002
kd = 0.9999995
y = []

while kc<0.99:
    y.append(kc)
    kc = kc**kd
ax1.plot(y)
#plt.xlim(0,1e+7)
plt.show()

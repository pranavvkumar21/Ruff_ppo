#!/usr/bin/env python3
c = 2e-10
d = 0.99999994

done = 0
counter = 0
while not done:
    c = c**d
    if c>0.98:
        done = 1
    if counter%1_000_000==0:
        print(c)
    counter +=1
print(counter)

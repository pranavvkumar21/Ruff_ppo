#!/usr/bin/env python3
c = 0.0000002
d = 0.9991
done = 0
counter = 0
while not done:
    c = c**d
    if c>0.98:
        done = 1
    if counter%1000==0:
        print(c)
    counter +=1
print(counter)

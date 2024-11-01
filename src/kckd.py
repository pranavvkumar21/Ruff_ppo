#!/usr/bin/env python3
c = 0.3
d = 0.999995

done = 0
counter = 0
while not done:
    c = c**d
    if c >= 0.99:
        done = 1
    if counter%1_000_000==0:
        print(c)
    if counter ==1_300_000:
        print("c: "+str(c) )
    counter +=1
print(counter)

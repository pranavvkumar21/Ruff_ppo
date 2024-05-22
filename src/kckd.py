#!/usr/bin/env python3
c = 2e-10
d = 0.999994

done = 0
counter = 0
while not done:
    c = c**d
    if counter == 2000000:
        done = 1
    if counter%1_000_000==0:
        print(c)
    if counter ==1_300_000:
        print("c: "+str(c) )
    counter +=1
print(counter)

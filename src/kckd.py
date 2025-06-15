#!/usr/bin/env python3
c = 0.005
d = 0.9995

done = 0
counter = 0
while not done:
    c = c**d
    if c >= 0.99:
        done = 1
    if counter%1000==0:
        print(f"c:{c}")
        #done=1
    if counter ==750_000:
        print("c: "+str(c) )
        #done=1
    counter +=1
print(counter)

#!/usr/bin/env python3
from ruff import *
from math import radians,degrees
id  = setup_world(3)
print(id)
STEPS = 1000
angle = [0]*12
angle = radians(45)
#angle[2] = radians(-90)
ruff_s = []
for i in id:
    print(i)
    ruff_s.append(ruff(i))
for step in range(STEPS):

    time.sleep(0.001)
    for ru in ruff_s:
        print(ru.id)
        ru.target_pos[2] = angle
        ru.move()
        ru.getjointinfo()
        #print(ru.joint_names)
        #print([degrees(i)//1 for i in ru.joint_position])
    p.stepSimulation()

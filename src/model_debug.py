#!/usr/bin/env python3
from ruff import *
from math import radians,degrees

client_mode = p.DIRECT
id  = setup_world(9,client_mode)
print(id)
STEPS = 1000
angle = [0]*12
angle = radians(45)
#angle[2] = radians(-90)
ruff_s = []
for i in id:
    print(i)
    ruff_s.append(ruff(i))
start_t = time.time()
for step in range(STEPS):

    #time.sleep(0.001)
    for ru in ruff_s:

        #print(ru.id)
        ru.get_state()
        ru.target_pos[2] = angle
        ru.move()
        ru.getjointinfo()
        #print(ru.joint_names)
        #print([degrees(i)//1 for i in ru.joint_position])
    p.stepSimulation()
    print(step)
end_t = time.time()
print(end_t-start_t)

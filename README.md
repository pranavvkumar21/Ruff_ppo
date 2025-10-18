#  Bio-Inspired Rhythmic Locomotion with Quadruped Robot using Proximal Policy Optimization Reinforcement Learning

In this project, we demonstrate how a rhythm generator can naturally stimulate period motor patterns within current reinforcement learning frameworks. The code is written in Python and uses the keras and pybullet simulator.


## Overview

The project uses a rhythm generator (RG) network to adjust the timing of phase transitions between the swing phase and the stance phase, and a pattern formation (PF) network to output motor commands for each leg of the robot. The similar structure also exists in the mammalian nervous system, where the RG network determines the flexor and extensor phase duration, and the PF network generates cyclical activation of the flexor and extensor motor neurons.

From an engineering perspective, the code encourages the robot to lift the corresponding foot high when the leg is in the swing phase or hold the corresponding foot firmly to the ground when the leg is in the stance phase. The cycle of the leg phases ensures the emergence of the animal-like rhythmic locomotion behaviors. With the proposed control architecture, we can focus on the main tasks of legged robots, such as forward movement and steering movement, etc.

# Checklist of Things to do

- [x]   Fix Reset spawn
- [x]   setup stable baselines3
- [x]   Fix target_pos action and set max torque limits
- [x]   Create Curriculum 
- [x]   Set up foot slip, stance, clear and zvel rewards
- [ ]   Setup joint vel, torque and policy smooth penalty
- [ ]   Setup RG_freq and RG_phase rewards
- [ ]   Create rewards config.yaml for controlling reward parameters and weights
- [ ]   Move pybullet code into legacy-v2 restructure the repo
- [ ]   Add random push to the robot
- [ ]   Set terrain difficulty curriculum
- [ ]   Add domain randomization
- [ ]   Try using State dependant exploration and squashing outputs instead of clipping


## References

Sheng, Jiapeng, et al. "Bio-Inspired Rhythmic Locomotion for Quadruped Robots." IEEE Robotics and Automation Letters 7.3 (2022): 6782-6789.

## Requirements
The following libraries are required to run the code:
(-) numpy
(-) tensorflow
(-) keras
(-) tensorflow_probability
(-) pybullet

You can install all the required libraries using the following command:
```
pip install numpy tensorflow keras tensorflow_probability pybullet
```
## Usage
To train the quadruped robot to generate bio-inspired rhythmic locomotion using proximal policy optimization (PPO) reinforcement learning, follow these steps:
1. Configure the environment, log file path and the model and the curriculum.
2. set the client_mode to p.GUI for rendering the simulation.
3. set LOAD to True if you want to use a pre-trained weights
4. Train the quadruped robot: Use the following command to start the training:
```
chmod +x ruff_train_ppo.py
./ruff_train_ppo.py
```

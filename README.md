# Bio-Inspired Rhythmic Locomotion with Quadruped Robot using Proximal Policy Optimization Reinforcement Learning

This project demonstrates how a rhythm generator (RG) can naturally stimulate periodic motor patterns within reinforcement learning frameworks. The system is implemented in Python using **Stable-Baselines3** and **Isaac Lab**.

## Overview

The model uses two coupled subsystems:

* **Rhythm Generator (RG)**: Adjusts timing of phase transitions between swing and stance phases.
* **Pattern Formation (PF)**: Generates motor commands for each leg.

This structure mimics the mammalian central pattern generator (CPG), where the RG defines flexor/extensor phase durations, and the PF produces rhythmic motor activations.

From an engineering standpoint, this approach helps the robot coordinate leg cycles for stable, animal-like locomotion. The control architecture focuses on core legged-robot tasks such as forward walking and steering.

## Checklist

- [ ] Add random push to the robot

- [ ] Set terrain difficulty curriculum

- [ ] Add domain randomization

## Requirements

This project depends on Isaac Lab and Stable-Baselines3. Install dependencies using:

```bash
conda env create -f env.yaml
conda activate ruff_env
```

If using pip:

```bash
pip install torch stable-baselines3 tabulate pyfiglet natsort
```

### Installing Isaac Lab and Isaac Sim

Follow the official guide. Key pip steps:

```bash
# PyTorch CUDA 12.8 build (per docs)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Isaac Lab with Isaac Sim via NVIDIA index
pip install "isaaclab[isaacsim,all]==2.2.0" --extra-index-url https://pypi.nvidia.com
```

Tip: Python 3.11 is required for Isaac Sim 5.x. Verify with:

```bash
isaacsim --help
```

## Configuration

All simulation and training parameters are defined in the **config** folder:

* `ruff_config.yaml` – Defines scene, environment, and training/evaluation parameters.
* `ruff_reward.yaml` – Defines reward weights and term parameters.

Modify these files to adjust terrain setup, curriculum progression, training iterations, and evaluation modes.

## Usage

Two CLI arguments control the runtime behavior:

```bash
--mode  : choose between 'train' or 'eval' (default: eval)
--load  : flag to load the latest saved model checkpoint
```

### Example commands

Run training:

```bash
python src/run_ruff.py --mode train
```

Resume training from last checkpoint:

```bash
python src/run_ruff.py --mode train --load
```

Evaluate trained model:

```bash
python src/run_ruff.py --mode eval --load
```

## References

Sheng, Jiapeng, et al. *"Bio-Inspired Rhythmic Locomotion for Quadruped Robots."* IEEE Robotics and Automation Letters, 7(3), 2022, pp. 6782–6789.

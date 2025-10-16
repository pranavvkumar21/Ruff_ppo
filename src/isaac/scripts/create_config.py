import json


config = {}
config["num_envs"] = 5
config["env_spacing"] = 4.0
config["sim_dt"] = 0.001
config["decimation"] = 10
config["max_episode_length"] = 1000
config["episode_length_s"] = 10
config["headless"] = True
config["livestream"] = 2
config["joint_names"] = joint_names = [
            "FL1_", "FR1_", "RL1_", "RR1_",
            "FL2_", "FR2_", "RL2_", "RR2_",
            "FL3_", "FR3_", "RL3_", "RR3_",
        ]


with open("../../config/config.json", "w") as f:
    json.dump(config, f, indent=4)
print("Config file created at ../../config/config.json")
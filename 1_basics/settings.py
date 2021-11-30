params = {}

# General Hyperparameters
params["gamma"] = 0.99

# Planning Hyperparameters
params["horizon"] = 10
params["simulations"] = 100

# RL Hyperparameters
params["alpha"] = 0.001

# DQN Hyperparameters
params["memory_capacity"] = 5000
params["warmup_phase"] = 2500
params["target_update_interval"] = 1000
params["minibatch_size"] = 32
params["epsilon_linear_decay"] = 1.0/params["memory_capacity"]
params["epsilon_min"] = 0.01

# AlphaZero Hyperparameters
params["az_warmup_phase"] = 10000
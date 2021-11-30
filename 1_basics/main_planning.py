from agents.agent import RandomAgent
import environments.rooms as rooms
import agents.planning as planning
import matplotlib.pyplot as plot
import sys
import numpy
from settings import params
from experiment import episode

algorithm_names = ["MCR", "MCTS"]
training_episodes = 10
algorithm = sys.argv[1].upper()
assert algorithm in algorithm_names,\
    "Unknown algorithm '{}', expected to be in {}".format(algorithm, algorithm_names)

# Domain setup
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms.mp4")
params["nr_actions"] = env.action_space.n
params["nr_input_features"] = env.observation_space.shape
params["env"] = env

# Agent setup
if algorithm == "MCR":
    agent = planning.MonteCarloRolloutPlanner(params)
if algorithm == "MCTS":
    agent = planning.MonteCarloTreeSearchPlanner(params)
print("Running {} ...".format(algorithm))
returns = [episode(agent, False, params, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

print("\nRunning Random agent ...")
random_return = numpy.mean([episode(RandomAgent(params), False, params, i) for i in range(training_episodes)])
plot.plot(x,y,label=algorithm,color="b")
plot.plot([0,training_episodes-1], [random_return, random_return], label="Random", color="r")
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.legend()
plot.show()

env.save_video()
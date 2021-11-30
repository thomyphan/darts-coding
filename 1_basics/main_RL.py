from agents.agent import RandomAgent
import environments.rooms as rooms
import agents.a2c as a2c
import agents.dqn as dqn
import agents.alpha_zero as alpha_zero
import agents.reinforcement_learning as rl
import matplotlib.pyplot as plot
import gym
import sys
import numpy
from settings import params
from experiment import episode

algorithm_names = ["QL", "DQN", "A2C", "AZ"]
training_episodes = int(sys.argv[1])
algorithm = sys.argv[2].upper()
assert algorithm in algorithm_names, \
    "Unknown algorithm '{}', expected to be in {}".format(algorithm, algorithm_names)
render_episode = len(sys.argv) > 3

# Domain setup
if render_episode:
    domain_names = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']
    gym_domain_name = sys.argv[3]
    assert gym_domain_name in domain_names, \
        "Unknown gym domain '{}', expected to be in {}".format(algorithm, algorithm_names)
    env = gym.make(gym_domain_name)
else:
    env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms.mp4")
params["nr_actions"] = env.action_space.n
params["nr_input_features"] = env.observation_space.shape
params["env"] = env

# Agent setup
if algorithm == "QL":
    agent = rl.QLearner(params)
if algorithm == "A2C":
    agent = a2c.A2CLearner(params)
if algorithm == "DQN":
    agent = dqn.DQNLearner(params)
if algorithm == "AZ":
    agent = alpha_zero.SimpleAlphaZero(params)
returns = [episode(agent, render_episode, params, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

print("\nRunning Random agent ...")
random_return = numpy.mean([episode(RandomAgent(params), False, params, i) for i in range(10)])
plot.plot(x,y,label=algorithm,color="b")
plot.plot([0,training_episodes-1], [random_return, random_return], label="Random", color="r")
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.legend()
plot.show()

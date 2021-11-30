import environments.rooms as rooms
import agents.agent as agent
import matplotlib.pyplot as plot
from settings import params
from experiment import episode

training_episodes = 10

# Domain setup
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms.mp4")
params["nr_actions"] = env.action_space.n
params["nr_input_features"] = env.observation_space.shape
params["env"] = env

# Agent setup
agent = agent.RandomAgent(params)
returns = [episode(agent, False, params, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x,y,label="Random",color="b")
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.legend()
plot.show()

env.save_video()
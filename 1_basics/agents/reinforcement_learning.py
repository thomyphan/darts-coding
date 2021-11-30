import numpy
from agents.agent import Agent
from agents.multi_armed_bandits import epsilon_greedy
        
"""
 Autonomous agent using SARSA.
"""
class QLearner(Agent):

    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = {}
        self.alpha = params["alpha"]
        self.epsilon = 1
        self.epsilon_linear_decay = params["epsilon_linear_decay"]
        self.epsilon_min = params["epsilon_min"]
        
    def Q(self, state):
        state = numpy.array2string(state)
        if state not in self.Q_values:
            self.Q_values[state] = numpy.zeros(self.nr_actions)
        return self.Q_values[state]

    def policy(self, state):
        pass
        # TODO
        
    def update(self, state, action, reward, next_state, done):
        pass
        # TODO
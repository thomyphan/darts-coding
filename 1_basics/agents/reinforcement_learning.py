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
        Q_val = self.Q(state)
        return epsilon_greedy(Q_val, None, self.epsilon)
        
    def update(self, state, action, reward, next_state, done):
        old_Q = self.Q(state)[action]
        new_Q = reward + self.gamma*max(self.Q(next_state))
        self.Q(state)[action] = (1-self.alpha)*old_Q + self.alpha*new_Q
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_linear_decay)
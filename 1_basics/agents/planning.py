import numpy
import copy
from agents.agent import Agent
from agents.multi_armed_bandits import UCB1

"""
 Autonomous agent using Monte Carlo Rollout Planning.
"""
class MonteCarloRolloutPlanner(Agent):

    def __init__(self, params):
        super(MonteCarloRolloutPlanner, self).__init__(params)
        self.env = params["env"]
        self.gamma = params["gamma"]
        self.horizon = params["horizon"]
        self.simulations = params["simulations"]
        
    def policy(self, state):
        pass
        # TODO

"""
 Represents a (state) node in Monte Carlo Tree Search.
"""
class MonteCarloTreeSearchNode:
    
    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.horizon = params["horizon"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = numpy.zeros(self.nr_actions)
        self.action_counts = numpy.zeros(self.nr_actions)
        self.children = []

    """
     Selects an action according to a Multi-armed Bandit strategy.
     @return the selected action.
    """
    def select(self, Q_values, action_counts):
        pass
        # TODO

    """
     Appends a new child node to self.children.
    """
    def expand(self):
        pass
        # TODO
        
    """
     Performs a rollout for self.horizon-depth time steps.
     @return the obtained discounted return.
    """
    def rollout(self, generative_model, depth):
        pass
        # TODO
        
    """
     Updates the Q-values of this node according to
     the observed discounted return and the selected action.
    """
    def backup(self, discounted_return, action):
        self.action_counts[action] += 1
        N = self.action_counts[action]
        self.Q_values[action] = (N-1)*self.Q_values[action] + discounted_return
        self.Q_values[action] /= N
        
    """
     Makes a final decision based on the currently learned Q-values.
     @return the action with the highest Q-value.
    """
    def final_decision(self):
        return numpy.argmax(self.Q_values)

    """
     Indicates if this node is still a leaf node.
     @return True if this leaf is not fully expanded yet. False otherwise.
    """
    def isLeaf(self):
        return len(self.children) < self.nr_actions

    """
     Performs a simulation step in this node.
     @return the discounted return observed from this node.
    """
    def simulate(self, generative_model, depth):
        if depth >= self.horizon:
            return 0
        if self.isLeaf():
            self.expand()
            selected_action = len(self.children) - 1 # Select action that leads to new child node
            _, reward, _, _ = generative_model.step(selected_action)
            return reward + self.gamma*self.simulate_with_rollout(generative_model, selected_action, depth)
        selected_action = self.select(self.Q_values, self.action_counts)
        return self.simulate_with_selection(generative_model, selected_action, depth)
            
    """
     Simulates and evaluates an action with a subsequent rollout.
     @return the discounted return observed from this node.
    """
    def simulate_with_rollout(self, generative_model, action, depth):
        return self.simulate_action(generative_model, action, depth, self.rollout)
        
    """
     Simulates and evaluates an action with a simulation at a child node.
     @return the discounted return observed from this node.
    """
    def simulate_with_selection(self, generative_model, action, depth):
        return self.simulate_action(generative_model, action, depth, self.children[action].simulate)
     
    """
     Simulates and evaluates an action with a subsequent evaluation function.
     @return the discounted return observed from this node.
    """     
    def simulate_action(self, generative_model, action, depth, eval_func):
        _, reward, done, _ = generative_model.step(action)
        delayed_return = 0
        if not done:
            delayed_return = eval_func(generative_model, depth+1)
        discounted_return = reward + self.gamma*delayed_return
        self.backup(discounted_return, action)
        return discounted_return
        
"""
 Autonomous agent using Monte Carlo Tree Search for Planning.
"""
class MonteCarloTreeSearchPlanner(Agent):

    def __init__(self, params):
        super(MonteCarloTreeSearchPlanner, self).__init__(params)
        self.params = params
        self.env = params["env"]
        self.simulations = params["simulations"]

    def policy(self, state):
        pass
        # TODO

    
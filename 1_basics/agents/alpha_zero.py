import numpy
import copy
from agents.planning import MonteCarloTreeSearchPlanner
from agents.a2c import A2CLearner
from agents.multi_armed_bandits import PUCB
from torch.distributions import Categorical
import torch
import torch.nn.functional as F

"""
 Represents a (state) node in Monte Carlo Tree Search.
"""
class SimpleAlphaZeroNode:
    
    def __init__(self, params, prior_probs, value):
        self.prior_probs = prior_probs
        self.value = value
        self.params = params
        self.gamma = params["gamma"]
        self.horizon = params["horizon"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = numpy.zeros(self.nr_actions)
        self.action_counts = numpy.zeros(self.nr_actions)
        self.rl_function = params["rl_function"]
        self.children = []

    """
     Selects an action according to a prior distribution.
     @return the selected action.
    """
    def select(self, Q_values, action_counts, prior_probs):
        return PUCB(Q_values, action_counts, prior_probs)

    """
     Appends a new child node to self.children.
    """
    def expand(self, state):
        prior_probs, value = self.rl_function(state)
        self.children.append(SimpleAlphaZeroNode(self.params, prior_probs, value))
        
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
     Compute final MCTS probs based on planning iterations.
     @return action probabilities proportional to the internal action counts.
    """
    def action_probs(self):
        total_action_count = 1.0*sum(self.action_counts)
        return self.action_counts/total_action_count

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
            self.expand(generative_model.state())
            selected_action = len(self.children) - 1 # Select action that leads to new child node
            _, reward, _, _ = generative_model.step(selected_action)
            return reward + self.gamma*self.value
        selected_action = self.select(self.Q_values, self.action_counts, self.prior_probs)
        return self.simulate_with_selection(generative_model, selected_action, depth)
        
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
class SimpleAlphaZero(MonteCarloTreeSearchPlanner):

    def __init__(self, params):
        super(SimpleAlphaZero, self).__init__(params)
        self.params = params
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.warmup_phase = params["az_warmup_phase"]
        self.nr_input_features = numpy.prod(params["nr_input_features"])
        self.transitions = []
        self.device = torch.device("cpu")
        self.rl_learner = A2CLearner(params)
        self.optimizer = self.rl_learner.optimizer
        def rl_function(state):
            prior_probs, value = self.rl_learner.predict([state])
            return prior_probs.detach().numpy()[0], value.detach().numpy()[0]      
        self.params["rl_function"] = rl_function
        self.transitions = []
        self.mcts_recommendations = None

    def policy(self, state):
        prior_probs, value = self.params["rl_function"](state)
        if self.warmup_phase <= 0:
            root = SimpleAlphaZeroNode(self.params, prior_probs, value)
            for _ in range(self.simulations):
                generative_model = copy.deepcopy(self.env)
                root.simulate(generative_model, depth=0)
            self.mcts_recommendations = root.action_probs()
            action = numpy.random.choice(range(self.nr_actions), p=self.mcts_recommendations)
        else:
            self.mcts_recommendations = numpy.ones(self.nr_actions)*1.0/self.nr_actions
            self.warmup_phase -= 1
            action = numpy.random.choice(range(self.nr_actions), p=prior_probs)
        return action

    """
     Performs a learning update of the currently learned policy and value function.
    """
    def update(self, state, action, reward, next_state, done):
        if self.warmup_phase > 0:
            self.rl_learner.update(state, action, reward, next_state, done)
        else:
            self.transitions.append((state, action, reward, next_state, self.mcts_recommendations, done))
            loss = None
            if done:
                states, _, rewards, _, mcts_recommendations, _ = tuple(zip(*self.transitions))
                discounted_returns = []
                R = 0
                for reward in reversed(rewards):
                    R = reward + self.gamma*R
                    discounted_returns.append(R)
                discounted_returns.reverse()
                discounted_returns = torch.tensor(discounted_returns, device=self.device, dtype=torch.float).detach()
                mcts_recommendations = torch.tensor(mcts_recommendations, device=self.device, dtype=torch.float).detach()

                # Calculate losses of policy and value function
                action_probs, state_values = self.rl_learner.predict(states)
                losses = []
                actions = torch.arange(0, self.nr_actions)
                for probs, mcts_probs, value, R in zip(action_probs, mcts_recommendations, state_values, discounted_returns):
                    m = Categorical(probs)
                    policy_loss = torch.stack([-m.log_prob(a) * mcts_probs[a] for a in actions]).sum()
                    value_loss = F.mse_loss(value, torch.tensor([R]))
                    losses.append(policy_loss + value_loss)
                loss = torch.stack(losses).mean()

                # Optimize joint loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.transitions.clear()
            return loss

    
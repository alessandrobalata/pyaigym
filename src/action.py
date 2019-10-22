import gym
import numpy as np


class Action:

    def __init__(self, action_space, epsilon, alpha):
        self.epsilon = epsilon
        self.alpha = alpha
        self.past_actions = []
        self.action_space = action_space
        self.current = None
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self.n = action_space.n
            self.cardinality = [action_space.n]

    def find_next(self, state, Q):
        self.past_actions.append(self.current)
        self.current = self._greedy_search(Q.eval(state, flag=True))

    def _greedy_search(self, f):
        # Determine next action - epsilon greedy strategy
        if np.random.random() < 1 - self.epsilon:
            # print('greedy action')
            return np.argmax(f)
        else:
            # print('random action')
            return self.action_space.sample()

    def go_confident(self):
        if self.epsilon > 0.01:
            self.epsilon *= (1 - self.alpha)


import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


class QFunction:

    def __init__(self, learning_rate, discount):
        self.past_values = []
        self.current = None
        self.memory = []
        self.learning_rate = learning_rate
        self.discount = discount
        self.batch_size = 100

    def initialize(self, state, action, q):
        if not isinstance(q, np.ndarray):
            self.current = 0 * np.random.uniform(low=-1, high=1, size=np.array(state.cardinality).tolist() +
                                                                      action.cardinality)
        else:
            self.current = q

    def eval(self, state, action=None, flag=False):
        if np.size(state) == 1:
            if action is None:
                return self.current[state, :]
            else:
                return self.current[state, action]
        elif np.size(state) == 2:
            if action is None:
                return self.current[state[0], state[1], :]
            else:
                return self.current[state[0], state[1], action]

    def update(self, current_state, current_action, reward):
        self.past_values.append(self.current)
        if np.size(current_state) == 1:
            self.current[current_state, current_action] += reward
        elif np.size(current_state) == 2:
            self.current[current_state[0], current_state[1], current_action] += reward

    def new(self, current_state, current_action, reward):
        self.past_values.append(self.current)
        if np.size(current_state) == 1:
            self.current[current_state, current_action] = reward
        elif np.size(current_state) == 2:
            self.current[current_state[0], current_state[1], current_action] = reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, state, action, reward, terminated, terminal_condition):
        # Allow for terminal states
        if terminated and eval(terminal_condition):
            self.new(state.bin_previous(), action.current, reward)
        # Adjust Q value for current state
        else:
            delta = self.learning_rate * (reward + self.discount *
                                          np.max(self.eval(state.bin_it())) -
                                          self.eval(state.bin_previous(), action.current))
            self.update(state.bin_previous(), action.current, delta)

    def plot_past_q(self, i):
        plt.figure()
        if np.size(self.past_values[i]) == 1:
            sns.heatmap(self.past_values[i][:, :])
        elif np.size(self.past_values[i]) == 2:
            sns.heatmap(self.past_values[i][:, :, 0])
        plt.show()

import numpy as np
import gym


class State:
    def __init__(self, state_space):
        self.past_states = []
        self.current, self.previous = None, None
        if isinstance(state_space, gym.spaces.box.Box):
            self.low = state_space.low
            self.high = state_space.high
            self.den = np.min([1E8 + self.high * 0, np.max([-1E8 + self.high * 0, (self.high - self.low)], axis=0)],
                              axis=0) / np.array([100] * len(self.high))
            self.cardinality = np.round(self.high * 0 + 100, 0).astype(int) + 1
            self.bin = True
        if isinstance(state_space, gym.spaces.discrete.Discrete):
            self.cardinality = [state_space.n]
            self.low = 0
            self.high = state_space.n - 1
            self.bin = False

    def bin_it(self):
        if self.bin:
            binned_state = (self.current - self.low) / self.den
            return np.round(binned_state, 0).astype(int)
        else:
            return self.current

    def bin_previous(self):
        if self.bin:
            binned_state = (self.previous - self.low) / self.den
            return np.round(binned_state, 0).astype(int)
        else:
            return self.previous

    def next(self, state):
        self.past_states.append(self.current)
        self.previous = self.current
        self.current = state

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque



class QAgent:

    def __init__(self, learning_rate, discount):
        self.past_values = []
        self.current = None
        self.memory = deque(maxlen=2000)
        self.learning_rate = learning_rate
        self.discount = discount
        self.batch_size = 100

    def initialize(self, state, action, w):
        self.current = Sequential()
        self.current.add(Dense(24, input_dim=len(state.cardinality), activation='relu'))
        self.current.add(Dense(24, activation='relu'))
        self.current.add(Dense(action.n, activation='linear'))
        self.current.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def eval(self, state, flag=False):
        if flag:
            return self.current.predict(state.reshape([1, 4]))[0]
        else:
            return self.current.predict(state.reshape([1, 4]))

    def update(self, state, target):
        return self.current.fit(state.reshape([1, 4]), target, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # if len(self.memory) > 10_000:
        #     del self.memory[np.random.randint(0, len(self.memory))]

    def replay(self, state, action, reward, terminated, terminal_condition):
        if terminated:
            for _ in range(self.batch_size):
                state, action, reward, next_state, done = self.memory[np.random.randint(0, len(self.memory))]
                target = reward
                if not done:
                    target = reward + self.discount * np.amax(self.eval(next_state, flag=True))
                target_f = self.eval(state)
                target_f[0][action] = target
                self.update(state, target_f)

class QEpisode:
    def __init__(self, env, state, action, Q):
        self.env = env
        self.state = state
        self.action = action
        self.Q = Q
        self.terminated, self.learning, self.discount = None, None, None
        self.collected_info = []

    def initialize(self, learning=0.01, discount=0.99):
        self.learning = learning
        self.discount = discount
        self.terminated = False
        self.state.next(self.env.reset())

    def play(self, show=False):
        tot_reward = 0
        step = 0
        while not self.terminated:
            step += 1
            if show:
                self.env.render()
            self.action.find_next(self.state.bin_it(), self.Q)
            new_location, reward, self.terminated, info = self.env.step(self.action.current, _print=False)
            self.state.next(new_location)
            self.collected_info.append(info)
            self.Q.remember(self.state.previous, self.action.current, reward, self.state.current, self.terminated)
            self.Q.replay(self.state, self.action, reward, self.terminated, self.env.terminal_condition)
            tot_reward += reward
        return tot_reward

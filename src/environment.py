import gym


class Environment:
    def __init__(self, environment):
        self.observation_space = None
        self.action_space = None
        self._env = self.make(environment)
        terminal_conditions = {'MountainCar-v0': 'state.current[0] >= 0.5',
                               'FrozenLake8x8-v0': 'False',
                               'DuplicatedInput-v0': 'False',
                               'CartPole-v1': 'False'}
        self.terminal_condition = terminal_conditions[environment]

    def make(self, environment: str = 'MountainCar-v0'):
        self._env = gym.make(environment)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        print(f'State space: {self.observation_space}')
        print(f'Action space: {self.action_space}')
        return self._env

    def reset(self):
        return self._env.reset()

    def step(self, action, _print=False):
        newstate, reward, terminated, info = self._env.step(action)
        # if reward != 0:
        #     print(reward)
        # if reward == 0 and terminated:
        #     reward = -1
        if _print:
            print(f'action = {action}')
            print(f'new state = {newstate}')
            print(f'reward = {reward}')
            print(f'info = {info}')
        if terminated:
            if _print:
                print('The episode has terminated')
        return newstate, reward, terminated, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

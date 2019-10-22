import numpy as np
import matplotlib.pyplot as plt


class Performance:
    def __init__(self):
        self.reward = None
        self.reward_list = []
        self.ave_reward_list = []
        self.ave_reward_ep_list = []

    def append(self, to_append):
        self.reward = to_append
        self.reward_list.append(to_append)

    def print(self, episode_number, custom='', reset=False):
        ave_reward = np.mean(self.reward_list)
        self.ave_reward_list.append(ave_reward)
        self.ave_reward_ep_list.append(episode_number)
        print(f'Episode {episode_number} Average Reward: {ave_reward} {custom}')
        if reset:
            self.reset()

    def reset(self):
        self.reward_list = []

    def plot(self):
        plt.figure()
        plt.plot(self.ave_reward_ep_list, self.ave_reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episodes')
        plt.show()

from src.environment import Environment
from src.action import Action
from src.q_learning.q_agent import QAgent
from src.q_learning.q_episode import QEpisode
from src.performace import Performance
from src.state import State
from src.q_learning.q_function import QFunction
import pickle


class QLearning:
    def __init__(self, environment: Environment, epsilon, epsilon_decay_rate, solver='table', q=None):
        self.state = State(environment.observation_space)
        self.action = Action(environment.action_space, epsilon, epsilon_decay_rate)
        if solver == 'table':
            self.q = QFunction(learning_rate, discount_rate)
        elif solver == 'NN_agent':
            self.q = QAgent(learning_rate, discount_rate)
        self.q.initialize(self.state, self.action, q)
        self.episode = QEpisode(environment, self.state, self.action, self.q)
        self.environment = environment
        self.performance = Performance()

    def train(self, learning, discount, no_of_episodes):
        for ep_no in range(no_of_episodes):
            self.episode.initialize(learning, discount)
            show = True if (ep_no + 1) % 50 == 0 else False  # if no_of_episodes - ep_no <= 2 else False
            reward = self.episode.play(show=show)
            self.performance.append(reward)
            if (ep_no + 1) % 10 == 0:
                self.performance.print(ep_no + 1, reset=True, custom=f'eps = {self.action.epsilon}')
            self.action.go_confident()
        self.environment.close()
        return self.performance


if __name__ == '__main__':
    # environment_name = 'MountainCar-v0'
    # environment_name = 'FrozenLake8x8-v0'
    environment_name = 'CartPole-v1'
    # environment_name = 'DuplicatedInput-v0'

    learning_rate = 0.001
    discount_rate = 0.95
    eps = .99
    alpha = 0.0005
    solver = 'NN_agent'

    episodes = 1000

    try:
        pickling_on = open(f"models/{environment_name}.pickle", "rb")
        q = pickle.load(pickling_on)
        pickling_on.close()
        print('loading completed')
    except FileNotFoundError:
        q = None

    env = Environment(environment_name)
    q_learn = QLearning(environment=env, epsilon=eps, epsilon_decay_rate=alpha, solver=solver)
    performance = q_learn.train(learning_rate, discount_rate, episodes)
    performance.plot()

    pickling_on = open(f"models/{environment_name}.pickle", "wb")
    pickle.dump(q_learn.q.current, pickling_on)
    pickling_on.close()

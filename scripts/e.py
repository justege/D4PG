from gym.utils import seeding
import gym
from gym import spaces
import numpy as np

import gym
from gym import spaces


class BasicEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30]
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,))
        self.state = np.array([0, 0, 0])

    def step(self, action):
        state = np.array([1, 1, 1])

        if (action[0] < 0.3) and (action[0] > 0.1) and (action[2] > 0.1) and (action[1] > 0.1) and (
                action[1] < 0.3) and (action[2] < 0.3):
            reward = 10
        elif (action[0] < 0.3) and (action[0] > 0.1) and (action[1] > 0.1) and (action[1] < 0.3) and (action[2] < 0.3):
            reward = 7
        elif (action[0] < 0.3) and (action[0] > 0.1) and (action[1] < 0.3) and (action[2] < 0.3):
            reward = 5
        elif (action[0] < 0.3) and (action[1] < 0.3) and (action[2] < 0.3):
            reward = 3
        elif (action[0] < 0.3) and (action[1] < 0.3):
            reward = 1
        else:
            reward = -1

        print(action)
        done = True
        info = {}
        return state, reward, done, info

    def reset(self):
        self.state = np.array([0, 0, 0])
        return np.array(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass


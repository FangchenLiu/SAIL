import gym
import numpy as np
from gym import spaces


class FourRoom(gym.Env):
    def __init__(self):
        self.n = 11
        self.map = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 4, 3, 3, 3, 3, 3, 1, 0,
            0, 1, 1, 4, 1, 0, 1, 1, 2, 1, 0,
            0, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0,
            0, 1, 1, 4, 1, 0, 0, 0, 2, 0, 0,
            0, 1, 1, 4, 1, 0, 1, 1, 2, 1, 0,
            0, 1, 1, 5, 5, 5, 5, 5, 2, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]).reshape((self.n, self.n))
        self.init()

    def init(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n*self.n,), dtype=np.float32)
        self.observation_space.n = self.n
        self.dx = [0, 1, 0, -1]
        self.dy = [1, 0, -1, 0]
        self.default_action = np.array([3, 2, 1, 0])
        self.inverse_default_action = np.array([1, 0, 3, 2])
        self.action_space = spaces.Discrete(len(self.dx))
        self.reset()

    def label2obs(self, x, y):
        a = np.zeros((self.n*self.n,))
        assert self.x < self.n and self.y < self.n
        a[x * self.n + y] = 1
        return a

    def get_obs(self):
        return self.label2obs(self.x, self.y)

    def reset(self):
        start = np.where(self.map == 2)
        assert len(start) == 2
        self.x, self.y = 5, 8
        self.done = False
        return self.get_obs()

    def set_xy(self, x, y):
        self.x = x
        self.y = y
        return self.get_obs()

    def compute_reward(self, prev_x, prev_y, action):
        info = {'is_success': False}
        done = False
        loc = self.map[prev_x, prev_y]
        assert loc > 0
        if loc < 2:
            reward = 0
        else:
            if action == self.default_action[loc-2]:
                reward = 1
            elif action == self.inverse_default_action[loc-2]:
                reward = -1
            else:
                reward = 0
        return reward, done, info

    def step(self, action):
        #assert not self.done
        nx, ny = self.x + self.dx[action], self.y + self.dy[action]
        info = {'is_success': False}
        #before = self.get_obs().argmax()
        if self.map[nx, ny]:
            reward, done, info = self.compute_reward(self.x, self.y, action)
            self.x, self.y = nx, ny
        else:
            #dis = (self.goal[0]-self.x)**2 + (self.goal[1]-self.y)**2
            #reward = -np.sqrt(dis)
            reward = 0
            done = False
        return self.get_obs(), reward, done, info

    def restore(self, obs):
        obs = obs.argmax()
        self.x = obs//self.n
        self.y = obs % self.n

    def inv_action(self, state, prev_state):
        x, y = state // self.n, state % self.n
        px, py = prev_state // self.n, prev_state % self.n
        dx = x - px
        dy = y - py
        if dx == 1 and dy == 0:
            return 1
        elif dx == -1 and dy == 0:
            return 3
        elif dy == 1 and dx == 0:
            return 0
        else:
            return 2


class FourRoom1(FourRoom):
    def __init__(self, seed=None, *args, **kwargs):
        FourRoom.__init__(self, *args, **kwargs)
        self.n = 11
        self.map = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 4, 3, 3, 3, 3, 3, 1, 0,
            0, 1, 1, 4, 1, 0, 1, 1, 2, 1, 0,
            0, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0,
            0, 1, 1, 4, 1, 0, 0, 0, 2, 0, 0,
            0, 1, 1, 4, 1, 0, 1, 1, 2, 1, 0,
            0, 1, 1, 5, 5, 5, 5, 5, 2, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]).reshape((self.n, self.n))
        self.init()

    def init(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n*self.n,), dtype=np.float32)
        self.observation_space.n = self.n
        self.dx = [0, 1, 0, -1, 0, 2, 0, -2]
        self.dy = [1, 0, -1, 0, 2, 0, -2, 0]
        self.default_action = np.array([3, 2, 1, 0, 7, 6, 5, 4])
        self.inverse_default_action = np.array([1, 0, 3, 2, 5, 4, 7, 6])
        self.action_space = spaces.Discrete(len(self.dx))
        self.reset()

    def compute_reward(self, prev_x, prev_y, action):
        info = {'is_success': False}
        done = False
        loc = self.map[prev_x, prev_y]
        assert loc > 0
        if loc < 2:
            reward = 0
        else:
            if action == self.default_action[loc-2]:
                reward = 1
            elif action == self.default_action[loc+2]:
                reward = 2
            elif action == self.inverse_default_action[loc-2]:
                reward = -1
            elif action == self.inverse_default_action[loc+2]:
                reward = -2
            else:
                reward = 0
        return reward, done, info

    def step(self, action):
        #assert not self.done
        nx, ny = max(0, self.x + self.dx[action]), max(0, self.y + self.dy[action])
        nx, ny = min(self.n-1, nx), min(self.n-1, ny)
        info = {'is_success': False}
        #before = self.get_obs().argmax()
        if self.map[nx, ny]:
            reward, done, info = self.compute_reward(self.x, self.y, action)
            self.x, self.y = nx, ny
        else:
            #dis = (self.goal[0]-self.x)**2 + (self.goal[1]-self.y)**2
            #reward = -np.sqrt(dis)
            reward = 0
            done = False
        return self.get_obs(), reward, done, info


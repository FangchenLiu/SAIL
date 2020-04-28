import gym
import numpy as np
from gym import spaces


class FourRoom(gym.Env):
    def __init__(self, seed=None, goal_type='fix_goal'):
        self.n = 11
        self.map = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]).reshape((self.n, self.n))
        self.goal_type = goal_type
        self.goal = None
        self.init()

    def init(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n*self.n,), dtype=np.float32)
        self.observation_space.n = self.n
        self.dx = [0, 1, 0, -1]
        self.dy = [1, 0, -1, 0]
        self.action_space = spaces.Discrete(len(self.dx))
        self.reset()

    def label2obs(self, x, y):
        a = np.zeros((self.n*self.n,))
        assert self.x < self.n and self.y < self.n
        a[x * self.n + y] = 1
        return a

    def get_obs(self):
        assert self.goal is not None
        return self.label2obs(self.x, self.y)

    def reset(self):
        '''
        condition = True
        while condition:
            self.x = np.random.randint(1, self.n)
            self.y = np.random.randint(1, self.n)
            condition = (self.map[self.x, self.y] == 0)
        '''
        self.x, self.y = 9, 9
        loc = np.where(self.map > 0.5)
        assert len(loc) == 2
        #if self.goal_type == 'random':
            # goal_idx = np.random.randint(len(loc[0]))
        if self.goal_type == 'fix_goal':
            goal_idx = 0
        else:
            raise NotImplementedError
        self.goal = loc[0][goal_idx], loc[1][goal_idx]
        self.done = False
        return self.get_obs()

    def set_xy(self, x, y):
        self.x = x
        self.y = y
        return self.get_obs()

    def step(self, action):
        #assert not self.done
        nx, ny = self.x + self.dx[action], self.y + self.dy[action]
        info = {'is_success': False}
        #before = self.get_obs().argmax()
        if self.map[nx, ny]:
            self.x, self.y = nx, ny
            #dis = (self.goal[0]-self.x)**2 + (self.goal[1]-self.y)**2
            #reward = -np.sqrt(dis)
            reward = -1
            done = False
        else:
            #dis = (self.goal[0]-self.x)**2 + (self.goal[1]-self.y)**2
            #reward = -np.sqrt(dis)
            reward = -1
            done = False
        if nx == self.goal[0] and ny == self.goal[1]:
            reward = 0
            info = {'is_success': True}
            done = self.done = True
        return self.get_obs(), reward, done, info

    def compute_reward(self, state, goal, info):
        state_obs = state.argmax(axis=1)
        goal_obs = goal.argmax(axis=1)
        reward = np.where(state_obs == goal_obs, 0, -1)
        return reward

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


    def bfs_dist(self, state, goal, order=True):
        #using bfs to search for shortest path
        visited = {key: False for key in range(self.n*self.n)}
        state_key = state.argmax()
        goal_key = goal.argmax()
        queue = []
        visited[state_key] = True
        queue.append(state_key)
        dist = [-np.inf] * (self.n*self.n)
        past = [-1] * (self.n*self.n)
        dist[state_key] = 0

        if order:
            act_order = range(4)
        else:
            act_order = range(3, 0, -1)

        while (queue):
            par = queue.pop(0)
            if par == goal_key:
                break
            x_par, y_par = par // self.n, par % self.n
            for action in act_order:
                x_child, y_child = x_par + self.dx[action], y_par + self.dy[action]
                child = x_child*self.n + y_child
                if self.map[x_child, y_child] == 0:
                    continue
                if visited[child] == False:
                    visited[child] = True
                    queue.append(child)
                    dist[child] = dist[par] + 1
                    past[child] = par

        state_action_pair = []
        cur_state = goal_key
        while cur_state is not state_key:
            prev_state = past[cur_state]
            prev_action = self.inv_action(cur_state, prev_state)
            x_prev, y_prev = prev_state // self.n, prev_state % self.n
            print(x_prev, y_prev)
            state_action_pair.append(np.hstack([self.label2obs(x_prev, y_prev), np.array((prev_action, ))]))
            cur_state = prev_state
        state_action_pair.reverse()
        state_action_pair.append(np.hstack([self.label2obs(goal_key // self.n, goal_key % self.n), np.array((prev_action, ))]))
        print(len(state_action_pair))
        return dist, state_action_pair

    def get_pairwise(self, state, target):
        dist = self.bfs_dist(state, target)
        return dist

    def all_states(self):
        states = []
        mask = []
        for i in range(self.n):
            for j in range(self.n):
                self.x = i
                self.y = j
                states.append(self.get_obs())
                if isinstance(states[-1], dict):
                    states[-1] = states[-1]['observation']
                mask.append(self.map[self.x, self.y] > 0.5)
        return np.array(states)[mask]

    def all_edges(self):
        A = np.zeros((self.n*self.n, self.n*self.n))
        mask = []
        for i in range(self.n):
            for j in range(self.n):
                mask.append(self.map[i, j] > 0.5)
                if self.map[i][j]:
                    for a in range(4):
                        self.x = i
                        self.y = j
                        t = self.step(a)[0]
                        if isinstance(t, dict):
                            t = t['observation']
                        self.restore(t)
                        A[i*self.n+j, self.x*self.n + self.y] = 1
        return A[mask][:, mask]

    def add_noise(self, start, goal, dist, alpha=0.1, order=False):

        if order:
            act_order = range(4)
        else:
            act_order = range(3, 0, -1)

        cur_state_id = start.argmax()
        goal_id = goal.argmax()
        new_seq = []
        while(cur_state_id != goal_id):
            x_cur, y_cur = cur_state_id // self.n, cur_state_id % self.n
            if np.random.randn() < alpha:
                cur_action = np.random.randint(4)
                nx, ny = x_cur+self.dx[cur_action], y_cur+self.dy[cur_action]
                new_seq.append(np.hstack([self.label2obs(x_cur, y_cur), np.array((cur_action, ))]))
                #print('state, action', (cur_state_id//self.n, cur_state_id%self.n), cur_action)
                if self.map[nx][ny] > 0.5:
                    cur_state_id = nx*self.n + ny
                else:
                    cur_state_id = cur_state_id
            else:
                dist_n = -np.inf
                cur_action = -1
                for action in act_order:
                    x_n, y_n = x_cur + self.dx[action], y_cur + self.dy[action]
                    if dist[x_n*self.n+y_n] > dist_n:
                        dist_n = dist[x_n*self.n+y_n]
                        cur_action = action
                    elif dist[x_n*self.n+y_n] == dist_n:
                        cur_action = np.random.choice(np.array([cur_action, action]))

                nx, ny = x_cur+self.dx[cur_action], y_cur+self.dy[cur_action]
                new_seq.append(np.hstack([self.label2obs(x_cur, y_cur), np.array((cur_action, ))]))
                #print('state, action', (cur_state_id//self.n, cur_state_id%self.n), cur_action)
                if self.map[nx][ny] > 0.5:
                    cur_state_id = nx*self.n + ny
                else:
                    cur_state_id = cur_state_id

        new_seq.append(np.hstack([self.label2obs(goal_id//self.n, goal_id%self.n), np.array((cur_action, ))]))
        return new_seq

class FourRoom1(FourRoom):
    def __init__(self, seed=None, *args, **kwargs):
        FourRoom.__init__(self, *args, **kwargs)
        self.n = 11
        self.map = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]).reshape((self.n, self.n))
        self.init()

    def init(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n*self.n,), dtype=np.float32)
        self.observation_space.n = self.n
        self.dx = [0, 1, 0, -1]
        self.dy = [2, 0, -2, 0]
        self.action_space = spaces.Discrete(len(self.dx))
        self.reset()

    def inv_action(self, state, prev_state):
        x, y = state // self.n, state % self.n
        px, py = prev_state // self.n, prev_state % self.n
        dx = x - px
        dy = y - py
        if dx == 1 and dy == 0:
            return 1
        elif dx == -1 and dy == 0:
            return 3
        elif dy == 2 and dx == 0:
            return 0
        else:
            return 2

    def step(self, action):
        #assert not self.done
        nx, ny = max(0, self.x + self.dx[action]), max(0, self.y + self.dy[action])
        nx, ny = min(self.n-1, nx), min(self.n-1, ny)
        info = {'is_success': False}
        #before = self.get_obs().argmax()
        if self.map[nx, ny]:
            self.x, self.y = nx, ny
            #dis = (self.goal[0]-self.x)**2 + (self.goal[1]-self.y)**2
            #reward = -np.sqrt(dis)
            reward = -1
            done = False
        else:
            #dis = (self.goal[0]-self.x)**2 + (self.goal[1]-self.y)**2
            #reward = -np.sqrt(dis)
            reward = -1
            done = False
        if nx == self.goal[0] and ny == self.goal[1]:
            reward = 0
            info = {'is_success': True}
            done = self.done = True
        return self.get_obs(), reward, done, info

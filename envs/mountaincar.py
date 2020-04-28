import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity = 0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = goal_velocity
        self.power = 0.001
        self.grav = 0.002
        print('power, grav', self.power, self.grav)
        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])
        '''
        modified
        '''
        self.pos_precision = 2
        self.vel_precision = 3
        self.pos_delta = 10 ** -self.pos_precision
        self.vel_delta = 10 ** -self.vel_precision
        self.pos_range = np.round(
            np.linspace(self.min_position, self.max_position, np.round((self.max_position - self.min_position) \
                                                                       / self.pos_delta + 1)),
            self.pos_precision)
        self.vel_range = np.round(
            np.linspace(-self.max_speed, self.max_speed, np.round((2 * self.max_speed) / self.vel_delta + 1)),
            self.vel_precision)
        
        self.test_pos_range =  np.round(
            np.linspace(-0.6, -0.4, np.round(0.2 / self.pos_delta + 1)),
            self.pos_precision)
        self.test_vel_range = np.round(
            np.linspace(-0.03, 0.03, np.round(0.06 / self.vel_delta + 1)),
            self.vel_precision)

        self.n_states = len(self.pos_range) * len(self.vel_range)
        self.n_test_states = len(self.test_pos_range) * len(self.test_vel_range)

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        #self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_id(self, state):
        if state is None:
            return self.n_states
        pos, vel = state
        if np.round(pos, self.pos_precision) > self.max_position:
            return self.n_states

        pos_id = np.where(self.pos_range == np.round(pos, self.pos_precision))[0][0]
        vel_id = np.where(self.vel_range == np.round(vel, self.vel_precision))[0][0]
        return pos_id * len(self.vel_range) + vel_id

    def visualize(self, dist):
        p = self.pos_range
        v = self.vel_range
        plot_V = np.zeros([len(p), len(v)])

        for i in range(len(p)):
            for j in range(len(v)):
                plot_V[i, j] = -dist[self.get_id((p[i], v[j]))]

        from matplotlib import pyplot as plt
        p = np.array(p)
        v = np.array(v)
        plot_V = plot_V.T
        extent = [np.amin(p), np.amax(p), np.amax(v), np.amin(v)]
        plt.imshow(plot_V, extent=extent, aspect='auto', cmap='hot', interpolation='nearest')
        plt.show()

    def step(self, action):

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -self.grav * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action[0],2)*0.1

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def vis_step(self, state, action):

        position = state[0]
        velocity = state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action[0],2)*0.1

        next_state = np.array([position, velocity])
        return next_state, reward, done, {}

    def reset(self):
        self.state = np.array([-0.5, 0])
        return np.array(self.state)

    #    def get_state(self):
    #        return self.state

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
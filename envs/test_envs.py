from utils.render import play
import gym
import envs
import envs.mujoco

if __name__ == '__main__':
    env = gym.make('DisableAnt-v0')
    play(env, None, None, video_path='disable_ant.avi', time_limit=200, device='cpu')
    env = gym.make('LightAnt-v0')
    play(env, None, None, video_path='light_ant.avi', time_limit=200, device='cpu')
    env = gym.make('HeavyAnt-v0')
    play(env, None, None, video_path='heavy_ant.avi', time_limit=200, device='cpu')
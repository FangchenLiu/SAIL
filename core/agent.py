from utils.replay_memory import OnlineMemory
from utils.torch import *
import math
import time

def collect_samples(env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size, update, trajectory, dtype=torch.float32):
    log = dict()
    memory = OnlineMemory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    if dtype == torch.float32:
        numpy_dtype = np.float32
    else:
        numpy_dtype = np.float64

    while num_steps < min_batch_size:
        state = env.reset()
        episode_steps = 0
        if running_state is not None:
            state = running_state(state, update=update)
        reward_episode = 0

        for t in range(10000):
            state_var = tensor(state).to(dtype).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(numpy_dtype)
            next_state, reward, done, _ = env.step(action)
            episode_steps += 1
            reward_episode += reward

            if running_state is not None:
                next_state = running_state(next_state, update=update)
                #print(next_state, running_state.rs.n)

            if episode_steps == env._max_episode_steps:
                mask = 1.
            else:
                mask = float(not done)

            if custom_reward is not None:
                reward = custom_reward(state, action, next_state, done)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            memory.push(state, action, reward, next_state, mask)
            if trajectory is not None:
                trajectory.push(state, action, reward, next_state, mask)

            if render:
                env.render()
            if done:
                if trajectory is not None:
                    trajectory.clear()
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_episodes
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    return memory, log

class Agent:

    def __init__(self, env, policy, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, update=True, dtype=torch.float32):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.update = update
        self.dtype = dtype

    def collect_samples(self, min_batch_size, trajectory=None):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        memory, log = collect_samples(self.env, self.policy, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, min_batch_size, self.update, trajectory, self.dtype)

        batch = memory.sample()
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log

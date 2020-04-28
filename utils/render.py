import cv2
import torch
import numpy as np

def play(env, policy, running_state=None, video_path="tmp.avi", time_limit=999, device='cpu'):
    out = None
    obs = env.reset()
    if running_state is not None:
        obs = running_state(obs, update=False)
    num = 0

    while True:
        img = env.unwrapped.render(mode='rgb_array')[:, :, ::-1].copy()
        if out is None:
            out = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (img.shape[1], img.shape[0]))
        out.write(img)
        if policy is not None:
            obs = torch.tensor(obs).float().unsqueeze(0).to(device)
            action = policy.select_action(obs)[0].detach().cpu().numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float32)
        else:
            action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if running_state is not None:
            obs = running_state(obs, update=False)
        if done:
            obs = env.reset()
        num += 1
        #assert not info['is_success']
        flag = True
        if not flag:
            print(num, info, rew, done, env.goal, action)
        if num == time_limit - 1:
            break
    env.close()

def play_action_seq(env, action_seq, video_path="tmp.avi", time_limit=999):
    out = None
    obs = env.reset()
    t = 0
    reward = 0
    while True:
        img = env.unwrapped.render(mode='rgb_array')[:, :, ::-1].copy()
        if out is None:
            out = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (img.shape[1], img.shape[0]))
        out.write(img)
        action = action_seq[t]
        obs, rew, done, info = env.step(action)
        reward+=rew
        t+=1
        if t > time_limit:
            print('accumulated reward', reward)
            break
    env.close()
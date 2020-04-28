from models.ppo_models import Value, Policy, DiscretePolicy
import pickle
import numpy as np
from utils import *
from utils.utils import adjust_lr
from core.common import estimate_advantages
from core.agent import Agent
import gym
from models.WGAN import W_Discriminator
from models.dynamics import ForwardModel, InverseModel
from tensorboardX import SummaryWriter
import datetime
import torch.autograd as autograd
from torch.distributions import Normal
from utils.utils import soft_update, hard_update
from models.VAE import VAE


class SoftBC_agent(object):

    def __init__(self, args, running_state=None):
        """environment"""
        self.env = gym.make(args.env_name)
        self.dtype = torch.float32
        self.args = args
        torch.set_default_dtype(self.dtype)
        self.state_dim = self.env.observation_space.shape[0]
        self.is_disc_action = len(self.env.action_space.shape) == 0
        self.action_dim = 1 if self.is_disc_action else self.env.action_space.shape[0]
        self.device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

        '''running normalizer'''
        if running_state is not None:
            self.running_state = running_state
        else:
            self.running_state = ZFilter((self.state_dim,), clip=5)

        """seeding"""
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.env.seed(args.seed)

        """define actor and critic"""
        if self.is_disc_action:
            self.policy_net = DiscretePolicy(self.state_dim, self.env.action_space.n)
        else:
            self.policy_net = Policy(self.state_dim, self.env.action_space.shape[0], log_std=args.log_std)
        self.value_net = Value(self.state_dim)
        self.goal_model = VAE(self.state_dim, latent_dim=128)
        self.inverse_model = InverseModel(self.state_dim*2, self.action_dim)
        self.discrim_net = W_Discriminator(self.state_dim, hidden_size=256)
        self.max_action = self.env.action_space.high[0]

        self.value_iter = self.args.value_iter
        self.policy_iter = 0

        to_device(self.device, self.policy_net, self.value_net, self.goal_model, \
                  self.inverse_model, self.discrim_net)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=args.policy_lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=args.value_lr)
        self.optimizer_vae = torch.optim.Adam(self.goal_model.parameters(), lr=args.model_lr)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.gan_lr, betas=(args.beta1, 0.99))
        self.optimizer_inverse = torch.optim.Adam(self.inverse_model.parameters(), lr=args.model_lr)

        self.save_path = '{}_softbc_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, \
                                                  args.beta)
        self.writer =SummaryWriter(log_dir='runs/'+self.save_path)

        self.trained = False
        self.trajectory = OnlineMemory()

        self.agent = Agent(self.env, self.policy_net, self.device, mean_action=True, custom_reward=self.expert_reward,
                               running_state=self.running_state, render=args.render, update=False)

    def expert_reward(self, state, action, next_state, done):
        coeff = 0.1 # to stablize
        state = torch.tensor(state, dtype=self.dtype).to(self.device)
        next_state = torch.tensor(next_state, dtype=self.dtype).to(self.device)
        if self.trained is False:
            return 0
        with torch.no_grad():
            reward = self.discrim_net(next_state[None, :]).item() - self.get_expert_mean()
            return reward*coeff

    def preprocess_running_state(self, expert_traj):
        expert_state_actions = expert_traj
        perm = np.arange(expert_state_actions.shape[0])
        np.random.shuffle(perm)
        expert_state_actions = expert_state_actions[perm].copy()
        expert_states = expert_state_actions[:, :self.state_dim]

        for i in range(expert_states.shape[0]):
            state = self.running_state(expert_states[i])
        # print(self.running_state.rs.n, self.running_state.rs.mean, self.running_state.rs.var)
        return self.running_state

    def pretrain_policy(self, epoches=150):
        state_tuples = torch.from_numpy(self.state_tuples).to(self.dtype).to(self.device)
        expert_states = state_tuples[:, :self.state_dim]
        expert_next_states = state_tuples[:, self.state_dim:self.state_dim*2]
        expert_actions = state_tuples[:, self.state_dim*2:]

        ''' during pretraining, we only optimize the mean, and keep the std of the policy to be a constant '''
        for i_epoch in range(epoches):
            idxs = np.arange(expert_states.shape[0])
            np.random.shuffle(idxs)
            num_batch = int(np.ceil(idxs.shape[-1] / self.args.optim_batch_size))
            for batch_num in range(num_batch):
                batch_idxs = idxs[batch_num * self.args.optim_batch_size : (batch_num + 1) * self.args.optim_batch_size]
                states = expert_states[batch_idxs].clone()
                mean_action, _, _ = self.policy_net.forward(states)

                next_states = self.goal_model.get_next_states(states)
                inverse_action = self.inverse_model.forward(states, next_states)

                policy_loss = ((inverse_action-mean_action)**2).mean()

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

            #if (i_epoch+1) % 100 == 0:
                #adjust_lr(self.optimizer_policy, 2.)
            self.writer.add_scalar('loss/pretraining', policy_loss, i_epoch)


    def pretrain_policy_l2(self, epoches=80):
        expert_state_actions = torch.from_numpy(self.expert_traj).to(self.dtype).to(self.device)
        batch_num = int(math.ceil(expert_state_actions.shape[0] / 256))
        for i in range(epoches):
            perm = np.arange(expert_state_actions.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).to(self.device)
            expert_state_actions = expert_state_actions[perm].clone()
            for b in range(batch_num):
                ind = slice(b * 256, min((b + 1) * 256, expert_state_actions.shape[0]))
                expert_sa_batch = expert_state_actions[ind]
                states = expert_sa_batch[:, :self.state_dim]
                actions = expert_sa_batch[:, self.state_dim:]
                policy_action, _, _ = self.policy_net.forward(states)
                self.optimizer_policy.zero_grad()
                loss = ((policy_action-actions)**2).mean(-1).mean(-1)
                loss.backward()
                self.optimizer_policy.step()
                if b == 0:
                    self.writer.add_scalar('loss/pretrain_policy_l2', loss, i)

    def update_params(self, batch, i_iter, total_steps):
        states = torch.from_numpy(np.stack(batch.state)).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.dtype).to(self.device)
        next_states = torch.from_numpy(np.stack(batch.next_state)).to(self.dtype).to(self.device)
        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.args.gamma, self.args.tau, self.device)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(states.shape[0] / self.args.optim_batch_size))
        for _ in range(self.args.optim_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).to(self.device)

            states, actions, next_states, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), next_states[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

            if i_iter > self.args.freeze_policy_iter:
                self.policy_iter = self.args.policy_iter

            for i in range(optim_iter_num):
                ind = slice(i * self.args.optim_batch_size, min((i + 1) * self.args.optim_batch_size, states.shape[0]))
                states_b, actions_b, next_states_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], next_states[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                self.meta_ppo_step(states_b, actions_b, returns_b,
                         advantages_b, fixed_log_probs_b, self.args.clip_epsilon, self.args.l2_reg, total_steps)


    def train(self):
        total_numsteps = 0
        for i_iter in range(self.args.max_iter_num):
            """generate multiple trajectories that reach the minimum batch_size"""
            to_device(torch.device('cpu'), self.policy_net, self.goal_model)
            batch, log = self.agent.collect_samples(self.args.min_batch_size)
            to_device(self.device, self.policy_net, self.goal_model)

            total_numsteps += log['num_steps']
            self.update_params(batch, i_iter, total_numsteps)
            self.train_models(batch, epoch=30)
            self.train_discrim(batch, epoch=30)

            if i_iter % self.args.log_interval == 0:
                print('{}\tT_sample {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}'.format(
                    i_iter, log['sample_time'], log['avg_c_reward'], log['avg_reward']))
                self.writer.add_scalar('reward/env', log['avg_reward'], total_numsteps)
                self.writer.add_scalar('reward/fake', log['avg_c_reward'], total_numsteps)

            if self.args.save_model_interval > 0 and (i_iter+1) % self.args.save_model_interval == 0:
                self.save_model()

            """clean up gpu memory"""
            torch.cuda.empty_cache()

    def split_data(self, expert_traj, state_pairs, state_tuples):
        self.expert_traj = expert_traj
        self.state_pairs = state_pairs
        idxs = np.arange(state_tuples.shape[0])
        np.random.shuffle(idxs)
        train_idxes = idxs[:(state_tuples.shape[0]*19)//20]
        test_idxes = idxs[(state_tuples.shape[0]*19)//20:]
        self.state_tuples = state_tuples[train_idxes]
        self.test_state_tuples = state_tuples[test_idxes]
        print('split train and validation', self.state_tuples.shape, self.test_state_tuples.shape)


    def train_discrim(self, batch, epoch=30):
        self.trained = True
        expert_data = torch.Tensor(self.expert_traj).to(self.device)
        expert_data = expert_data[:, :self.state_dim]
        imitator_data = torch.from_numpy(np.stack(batch.state)).to(self.dtype).to(self.device)
        #print(imitator_data.shape)
        for _ in range(epoch):
            self.optimizer_discrim.zero_grad()
            self.discrim_net.zero_grad()
            D_real = self.discrim_net(expert_data)
            D_real = D_real.mean()
            D_fake = self.discrim_net(imitator_data)
            D_fake = D_fake.mean()
            loss = -D_real + D_fake
            loss.backward()
            # train with gradient penalty
            gradient_penalty = self.calc_gradient_penalty(expert_data.data, imitator_data.data)
            gradient_penalty.backward()
            self.wasserstein = D_real - D_fake
            #print('wasserstein distance', self.wasserstein.item())
            self.optimizer_discrim.step()
        print('final wasserstein distance', self.wasserstein.item())
        return self.wasserstein.item()

    def get_expert_mean(self):
        expert_data = torch.Tensor(self.expert_traj).to(self.device)
        expert_data = expert_data[:, :self.state_dim]
        expert_mean = self.discrim_net(expert_data).mean().item()
        return expert_mean

    def train_models(self, batch, epoch):
        states = torch.from_numpy(np.stack(batch.state)).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.dtype).to(self.device)
        next_states = torch.from_numpy(np.stack(batch.next_state)).to(self.dtype).to(self.device)
        self.inverse_model.train(states, next_states, actions, self.optimizer_inverse, epoch=epoch, batch_size=self.args.optim_batch_size)

    def pretrain_vae(self, iter=200, epoch=2, lr_decay_rate=50):
        state_tuples = torch.from_numpy(self.state_tuples).to(self.dtype).to(self.device)
        s, t, action = state_tuples[:, :self.state_dim], state_tuples[:, self.state_dim:2*self.state_dim], \
                       state_tuples[:, 2*self.state_dim:]

        state_tuples_test = torch.from_numpy(self.test_state_tuples).to(self.dtype).to(self.device)
        s_test, t_test, action_test = state_tuples_test[:, :self.state_dim], state_tuples_test[:, self.state_dim:2 * self.state_dim], \
                       state_tuples_test[:, 2 * self.state_dim:]


        for i in range(1, iter + 1):
            loss = self.goal_model.train(s, t, epoch=epoch, optimizer=self.optimizer_vae, \
                                         batch_size=self.args.optim_batch_size, beta=self.args.beta)
            next_states = self.goal_model.get_next_states(s_test)
            val_error = ((t_test - next_states) ** 2).mean()
            self.writer.add_scalar('loss/vae', loss, i)
            self.writer.add_scalar('valid/vae', val_error, i)

            if i % lr_decay_rate == 0:
                adjust_lr(self.optimizer_vae, 2.)

    def pretrain_dynamics_with_l2(self, policy_epoch=50, iter=200, epoch=2, lr_decay_rate=50):
        '''
            designed for cross-morphology
            use l2 to pretrain policy
            collecting data with pretrained policy to train dynamics model
        '''
        self.pretrain_policy_l2(epoches=policy_epoch)
        memory_bc = self.warm_up(steps=50000, use_policy=True)
        state_tuples = torch.from_numpy(memory_bc).to(self.dtype).to(self.device)
        s, t, action = state_tuples[:, :self.state_dim], state_tuples[:, self.state_dim:2*self.state_dim], \
                       state_tuples[:, 2*self.state_dim:]

        # train the inverse model
        for i in range(1, iter + 1):
            loss = self.inverse_model.train(s, t, action, self.optimizer_inverse, epoch=epoch,
                                     batch_size=self.args.optim_batch_size)
            self.writer.add_scalar('loss/inverse', loss, i)

            if i % lr_decay_rate == 0:
                adjust_lr(self.optimizer_inverse, 2.)

    def pretrain_dynamics_with_demo(self, iter=200, epoch=2, lr_decay_rate=50):
        '''
        designed for normal setting
        use demo to train the dynamics model
        '''
        state_tuples = torch.from_numpy(self.state_tuples).to(self.dtype).to(self.device)
        s, t, action = state_tuples[:, :self.state_dim], state_tuples[:, self.state_dim:2*self.state_dim], \
                       state_tuples[:, 2*self.state_dim:]
        state_tuples_test = torch.from_numpy(self.test_state_tuples).to(self.dtype).to(self.device)
        s_test, t_test, action_test = state_tuples_test[:, :self.state_dim], state_tuples_test[:,
                                                                             self.state_dim:2 * self.state_dim], \
                                      state_tuples_test[:, 2 * self.state_dim:]

        # train the inverse model
        for i in range(1, iter + 1):
            loss = self.inverse_model.train(s, t, action, self.optimizer_inverse, epoch=epoch, batch_size=self.args.optim_batch_size)
            pred_acion = self.inverse_model.forward(s_test, t_test)
            val_error = ((pred_acion - action_test) ** 2).mean()
            self.writer.add_scalar('loss/inverse', loss, i)
            self.writer.add_scalar('valid/inverse', val_error, i)
            if i % lr_decay_rate == 0:
                adjust_lr(self.optimizer_inverse, 2.)


    def meta_ppo_step(self, states, actions, returns, advantages, fixed_log_probs, clip_epsilon, l2_reg, total_step, lam1=0.001, lam2=0.01):
        """update critic"""
        for _ in range(self.value_iter):
            values_pred = self.value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
            # weight decay
            for param in self.value_net.parameters():
                value_loss += param.pow(2).sum() * l2_reg
            self.optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 5)
            self.optimizer_value.step()
            self.writer.add_scalar('loss/value_loss', value_loss, total_step)

        for _ in range(self.policy_iter):
            """update policy"""
            log_probs = self.policy_net.get_log_prob(states, actions)
            ratio = torch.exp(log_probs - fixed_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_surr = -torch.min(surr1, surr2).mean()
            '''compute kl'''
            mean_action, action_log_std, action_std = self.policy_net.forward(states)
            next_states = self.goal_model.get_next_states(states)
            inverse_action = self.inverse_model.forward(states, next_states)

            # lam2 is a hyper-parameter, positive relevant to sigma^(-2) of action prior
            policy_loss = -lam2 * action_log_std.sum() + (action_std**2).sum() + ((inverse_action-mean_action)**2).mean()
            policy_loss += lam1 * policy_surr # balance the policy surrogate

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
            self.optimizer_policy.step()
            self.writer.add_scalar('loss/policy_loss', policy_loss, total_step)

    def warm_up(self, steps=5000, use_policy=False):
        memory = []
        t = 0
        if use_policy is True:
            to_device(torch.device('cpu'), self.policy_net)

        while t < steps:
            obs = self.env.reset()
            if self.running_state is not None:
                obs = self.running_state(obs, update=False)
            while True:
                t += 1
                if use_policy is True:
                    state_var = tensor(obs).to(self.dtype).unsqueeze(0)
                    action = self.policy_net.select_action(state_var)[0].numpy()
                else:
                    action = self.env.action_space.sample()

                next_obs, reward, done, _ = self.env.step(action)
                if self.running_state is not None:
                    next_obs = self.running_state(next_obs, update=False)
                transition = np.concatenate([obs, next_obs, action])
                memory.append(transition)
                if done:
                    break
                obs = next_obs
        memory = np.stack(memory)
        if use_policy is True:
            to_device(self.device, self.policy_net)
        return memory

    def eval(self, num=10):
        t = 0
        accu_reward = 0
        to_device(torch.device('cpu'), self.policy_net)
        while t < num:
            obs = self.env.reset()
            if self.running_state is not None:
                obs = self.running_state(obs, update=False)
            while True:
                state_var = tensor(obs).to(self.dtype).unsqueeze(0)
                action = self.policy_net(state_var)[0][0].detach().numpy()
                next_obs, reward, done, _ = self.env.step(action)
                accu_reward += reward
                if self.running_state is not None:
                    next_obs = self.running_state(next_obs, update=False)
                if done:
                    t += 1
                    break
                obs = next_obs
        to_device(self.device, self.policy_net)
        print('accumulated reward', accu_reward/num)
        return accu_reward/num

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(fake_data.shape[0], 1)
        idx = np.random.randint(0, len(real_data), fake_data.shape[0])
        real_data_b = real_data[idx]

        alpha = alpha.expand(real_data_b.size())
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data_b + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(self.device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.discrim_net(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.args.lam
        return gradient_penalty

    def save_model(self):
        to_device(torch.device('cpu'), self.inverse_model, self.policy_net, self.value_net, self.goal_model)
        print('saving models')
        torch.save([self.inverse_model.state_dict(), self.policy_net.state_dict(), self.value_net.state_dict(),
                    self.goal_model.state_dict()], assets_dir()+'/learned_models/{}_{}_models.pt'.format(self.args.env_name, str(self.args.beta)))
        to_device(self.device, self.inverse_model, self.policy_net, self.value_net, self.goal_model)

    def load_model(self):
        model_path = assets_dir()+'/learned_models/{}_{}_models.pt'.format(self.args.env_name, str(self.args.beta))
        print('load model from', model_path)
        to_device(torch.device('cpu'), self.inverse_model, self.policy_net, self.value_net, self.goal_model)
        pretrained_dict = torch.load(model_path)
        self.inverse_model.load_state_dict(pretrained_dict[0])
        self.policy_net.load_state_dict(pretrained_dict[1])
        self.value_net.load_state_dict(pretrained_dict[2])
        self.goal_model.load_state_dict(pretrained_dict[3])
        to_device(self.device, self.inverse_model, self.policy_net, self.value_net, self.goal_model)

    def save_replay(self):
        from utils.render import play
        to_device(torch.device('cpu'), self.policy_net)
        video_path = self.save_path+'.avi'
        play(self.env, self.policy_net, self.running_state, video_path=video_path, time_limit=1000, device='cpu')
        to_device(self.device, self.policy_net)

import datetime
import gym
import itertools
from agents.sac_agent import SAC_agent
from utils import *
import  argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch GAIL example')
    parser.add_argument('--env-name', default="Hopper-v2",
                        help='name of the environment to run')
    parser.add_argument('--policy', default="Gaussian",
                        help='algorithm to use: Gaussian | Deterministic')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default:True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Temperature parameter α automaically adjusted.')
    parser.add_argument('--seed', type=int, default=456, metavar='N',
                        help='random seed (default: 456)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num-steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden-size', type=int, default=400, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates-per-step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start-steps', type=int, default=300, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target-update-interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay-size', type=int, default=1e6, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='run on CUDA (default: False)')
    parser.add_argument('--actor-path', type=str, default='assets/learned_models/sac_actor_Hopper-v2_1', help='actor resume path')
    parser.add_argument('--critic-path', type=str, default='assets/learned_models/sac_critic_Hopper-v2_1', help='critic resume path')

    args = parser.parse_args()

    return args

args = get_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
state_dim = env.observation_space.shape[0]
agent = SAC_agent(env, env.observation_space.shape[0], env.action_space, args, running_state=None)
agent.load_model(actor_path=args.actor_path, critic_path=args.critic_path)
agent.save_expert_traj(max_step=50000)
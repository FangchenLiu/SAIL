import argparse
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from utils.utils import generate_pairs, process_expert_traj, generate_tuples, adjust_lr
from agents.soft_bc_agent import SoftBC_agent
from utils.utils import normalize_states, normalize_expert_traj

def get_args():
    parser = argparse.ArgumentParser(description='SAIL arguments')
    parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--expert-traj-path', metavar='G',
                        help='path of the expert trajectories')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment')
    parser.add_argument('--log-std', type=float, default=-5.0, metavar='G',
                        help='log std for the policy (default: -1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')

    parser.add_argument('--policy-lr', type=float, default=3e-4, metavar='G',
                        help='learning rate for policy networks')
    parser.add_argument('--value-lr', type=float, default=3e-4, metavar='G',
                        help='learning rate for value networks')
    parser.add_argument('--model-lr', type=float, default=3e-4, metavar='G',
                        help='learning rate for forward/inverse/vae')


    parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                        help='clipping epsilon for PPO')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size per PPO update (default: 2048)')
    parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                        help='maximal number of main iterations (default: 500)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--save-model-interval', type=int, default=50, metavar='N',
                        help="interval between saving model (default: 0, means don't save)")
    parser.add_argument('--freeze-policy-iter',type=int, default=20,
                        help="iteration to freeze pretrained policy, so that value network can be more stable")
    parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
    parser.add_argument('--rnd-epoch', type=int, default=4000, metavar='N')
    parser.add_argument('--running-state', type=int, default=1, metavar='N')
    parser.add_argument('--optim-epochs', type=int, default=10, metavar='N')
    parser.add_argument('--optim-batch-size', type=int, default=128, metavar='N')

    parser.add_argument('--load-running-state', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.005, help='beta VAE coefficient')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume pretrained models')
    parser.add_argument('--transfer', action='store_true', default=False,
                        help='if imitator and expert is different, then this should be true')

    '''
    hyperparameters for wgan
    '''
    parser.add_argument('--gan-lr', type=float, default=3e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=1.)
    parser.add_argument('--gan-batch-size', type=int, default=256)

    parser.add_argument('--value-iter', type=int, default=2)
    parser.add_argument('--policy-iter', type=int, default=1)
    args = parser.parse_args()
    return args
# load trajectory

if __name__ == '__main__':
    args = get_args()
    dtype = torch.float32
    torch.set_default_dtype(dtype)


    expert_traj_raw = pickle.load(open(args.expert_traj_path, "rb")) # list of expert trajectories
    bc_agent = SoftBC_agent(args)

    if isinstance(expert_traj_raw, np.ndarray):
        expert_traj_raw_list = []
        for i in range(len(expert_traj_raw)):
            expert_traj_raw_list.append(expert_traj_raw[i])
        expert_traj_raw = expert_traj_raw_list

    expert_traj = process_expert_traj(expert_traj_raw)
    state_pairs = generate_pairs(expert_traj_raw, bc_agent.state_dim, size_per_traj=1000, max_step=1)
    state_tuples = generate_tuples(expert_traj_raw, bc_agent.state_dim)

    running_state = bc_agent.preprocess_running_state(expert_traj)
    expert_traj = normalize_expert_traj(running_state, expert_traj, bc_agent.state_dim)
    state_pairs = normalize_states(running_state, state_pairs, bc_agent.state_dim)
    state_tuples = normalize_states(running_state, state_tuples, bc_agent.state_dim)


    bc_agent.split_data(expert_traj, state_pairs, state_tuples)

    if args.resume is False:
        bc_agent.pretrain_vae()
        if args.transfer is True:
            bc_agent.pretrain_dynamics_with_l2()
        else:
            print('pretrain with demo')
            bc_agent.pretrain_dynamics_with_demo()
        bc_agent.save_model()
        bc_agent.pretrain_policy()
        bc_agent.train()
    else:
        bc_agent.load_model()
        bc_agent.pretrain_policy(epoches=250)
        bc_agent.save_model()
        bc_agent.train()

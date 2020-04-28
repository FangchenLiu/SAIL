import argparse

def get_sac_args():
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
    parser.add_argument('--replay-size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='run on CUDA (default: False)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='run on CUDA (default: False)')
    parser.add_argument('--model-path', type=str, default='learned_models/sac')
    parser.add_argument('--expert-traj-path', metavar='G',
                        help='path of the expert trajectories')
    parser.add_argument('--rnd-epoch', type=int, default=600,
                        help='path of the expert trajectories')
    args = parser.parse_args()

    return args
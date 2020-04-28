import math
import torch
import numpy as np


def gen_dataset(expert_traj_raw, batch_size):
    length = len(expert_traj_raw)
    perm = np.arange(length)
    np.random.shuffle(perm)
    expert_traj = expert_traj_raw[perm].copy()
    while True:
        for i in range(len(expert_traj) // batch_size):
            yield expert_traj[i * batch_size:(i + 1) * batch_size]

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def fig2rgb_array(fig, expand=False):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)

from scipy import linalg

def compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.
    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type in 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol

def process_expert_traj(expert_traj_raw):
    expert_traj = []
    for i in range(len(expert_traj_raw)):
        for j in range(len(expert_traj_raw[i])):
            expert_traj.append(expert_traj_raw[i][j])
    expert_traj = np.stack(expert_traj)
    #print('here',  expert_traj.shape)
    return expert_traj

def generate_pairs(expert_traj_raw, state_dim, size_per_traj, max_step=6):
    '''
    generate state pairs (s, s_t)
    note that s_t can be multi-step future (controlled by max_step)
    '''
    pairs = []
    for i in range(len(expert_traj_raw)):
        traj = expert_traj_raw[i]
        if len(traj) == 0:
            continue
        start = np.random.randint(0, len(traj), size_per_traj)
        step = np.random.randint(1, max_step+1, size_per_traj)
        end = np.minimum(start+step, len(traj)-1)
        start_state, end_state = traj[start], traj[end]

        final_dim = state_dim*2
        state_pairs = np.concatenate([start_state[:, :state_dim], end_state[:, :state_dim]], axis=1)

        pairs.append(state_pairs)
    pairs = np.stack(pairs).reshape(-1, final_dim)
    return pairs

def generate_tuples(expert_traj_raw, state_dim):
    '''
    generate transition tuples (s, s', a) for training
    '''
    expert_traj = []
    for i in range(len(expert_traj_raw)):
        for j in range(len(expert_traj_raw[i])):
            if j < len(expert_traj_raw[i])-1:
                state_action = expert_traj_raw[i][j]
                next_state = expert_traj_raw[i][j+1][:state_dim]
                transitions = np.concatenate([state_action[:state_dim], next_state, state_action[state_dim:]], axis=-1)
                expert_traj.append(transitions)
    expert_traj = np.stack(expert_traj)
    return expert_traj

def normalize_expert_traj(running_state, expert_traj, state_dim):
    '''
    normalize the demonstration data by the state normalizer
    '''
    traj = []
    for i in range(len(expert_traj)):
        state = expert_traj[i, :state_dim]
        rest = expert_traj[i, state_dim:]
        state = running_state(state, update=False)
        tuple = np.concatenate([state, rest], axis=-1)
        traj.append(tuple)
    traj = np.stack(traj)
    return traj

def normalize_states(running_state, state_pairs, state_dim):
    '''
    normalize the state pairs/tuples by state normalizer
    '''
    traj = []
    for i in range(len(state_pairs)):
        state = state_pairs[i, :state_dim]
        next_state = state_pairs[i, state_dim:state_dim*2]
        rest = state_pairs[i, state_dim*2:]
        state = running_state(state, update=False)
        next_state = running_state(next_state, update=False)
        tuple = np.concatenate([state, next_state, rest], axis=-1)
        traj.append(tuple)
    traj = np.stack(traj)
    return traj


def adjust_lr(optimizer, scale):
    print('=========adjust learning rate================')
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / scale
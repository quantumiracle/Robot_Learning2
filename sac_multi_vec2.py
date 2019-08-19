import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import trange

from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from wrapper2 import make_vec_env


class ReplayBuffer(object):
    def __init__(self, size, device):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._device = device

    def __len__(self):
        return len(self._storage)

    def push(self, *args):
        if self._next_idx >= len(self._storage):
            self._storage.append(args)
        else:
            self._storage[self._next_idx] = args
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        b_o, b_a, b_r, b_o_, b_d = [], [], [], [], []
        for i in idxes:
            o, a, r, o_, d = self._storage[i]
            b_o.append(o)
            b_a.append(a)
            b_r.append(r)
            b_o_.append(o_)
            b_d.append(float(d))
        res = (
            torch.from_numpy(np.asarray(b_o)).to(self._device).float(),
            torch.from_numpy(np.asarray(b_a)).to(self._device).float(),
            torch.from_numpy(np.asarray(b_r)).unsqueeze(1).to(self._device).float(),
            torch.from_numpy(np.asarray(b_o_)).to(self._device).float(),
            torch.from_numpy(np.asarray(b_d)).unsqueeze(1).to(self._device).float(),
        )
        return res

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indexes = range(len(self._storage))
        idxes = [random.choice(indexes) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3,
                 log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample()
        action_0 = torch.tanh(mean + std * z.to(
            device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.from_numpy(state.astype(np.float32)).to(device)
            mean, log_std = self.forward(state)
            std = log_std.exp()

            normal = Normal(0, 1)
            z = normal.sample(std.size()).to(device)
            action = self.action_range * torch.tanh(mean + std * z)

            action = ((self.action_range * mean) if deterministic else action).cpu().numpy()
            return action

    def sample_action(self, sample_number):
        a = torch.Tensor(sample_number, self.num_actions).float().uniform_(-1, 1)
        return self.action_range * a.numpy()


if __name__ == '__main__':
    """Meta"""
    use_cuda = True
    num_workers = mp.cpu_count() * 2
    menv = mp.cpu_count()
    num_workers = 4
    menv = 2
    state_dim = 17
    action_dim = 4
    action_range = 0.1
    replay_buffer_size = 1e6
    max_timesteps = 1e7
    max_steps = 30
    explore_steps = 0  # for random action sampling in the beginning of training
    batch_size = 128 * int(menv ** 0.5)
    warm_start = batch_size * 1
    update_itr = 1
    action_itr = 3
    AUTO_ENTROPY = True
    DETERMINISTIC = False
    hidden_dim = 512
    model_path = './model/sac_v2_multi_vec'
    soft_q_lr = 3e-4
    policy_lr = 3e-4
    alpha_lr = 3e-4
    soft_q_criterion1 = nn.MSELoss()
    soft_q_criterion2 = nn.MSELoss()
    save_interval = 10000
    reward_scale = 10.0
    gamma = 0.99
    soft_tau = 1e-2

    """Init"""
    env = make_vec_env(num_workers, max_steps, menv)
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    replay_buffer = ReplayBuffer(replay_buffer_size, device)
    soft_q_net1 = nn.DataParallel(SoftQNetwork(state_dim, action_dim, hidden_dim).to(device))
    soft_q_net2 = nn.DataParallel(SoftQNetwork(state_dim, action_dim, hidden_dim).to(device))
    target_soft_q_net1 = nn.DataParallel(SoftQNetwork(state_dim, action_dim, hidden_dim).to(device))
    target_soft_q_net2 = nn.DataParallel(SoftQNetwork(state_dim, action_dim, hidden_dim).to(device))
    policy_net = nn.DataParallel(
        PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device))
    log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
    print('Soft Q Network (1,2): ', soft_q_net1)
    print('Policy Network: ', policy_net)
    soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
    soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    alpha_optimizer = optim.Adam([log_alpha], lr=alpha_lr)

    """Sync target network"""
    for target_param, param in zip(target_soft_q_net1.parameters(), soft_q_net1.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(target_soft_q_net2.parameters(), soft_q_net2.parameters()):
        target_param.data.copy_(param.data)

    """Start training"""
    o = env.reset()
    rewards = []
    max_iter = int(max_timesteps // menv)
    for eps in trange(1, max_iter + 1):
        if eps > explore_steps:
            a = policy_net.module.get_action(o, deterministic=DETERMINISTIC)
        else:
            a = policy_net.module.sample_action(menv)
        o_, r, d, info = env.step(a)
        for i in range(len(o)):
            replay_buffer.push(o[i], a[i], r[i], o_[i], d[i])
        o = o_
        if len(replay_buffer) > warm_start:
            for i in range(update_itr * int(menv ** 0.5)):
                state, action, reward, next_state, done = replay_buffer.sample(batch_size)

                predicted_q_value1 = soft_q_net1(state, action)
                predicted_q_value2 = soft_q_net2(state, action)
                new_action, log_prob, z, mean, log_std = policy_net.module.evaluate(state)
                new_next_action, next_log_prob, _, _, _ = policy_net.module.evaluate(next_state)
                # normalize with batch mean and std
                reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)
                # Updating alpha wrt entropy
                # alpha = 0.0
                # trade-off between exploration (max entropy) and exploitation (max Q)
                if AUTO_ENTROPY is True:
                    alpha_loss = -(log_alpha * (log_prob - 1.0 * action_dim).detach()).mean()
                    # print('alpha loss: ',alpha_loss)
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    alpha = log_alpha.exp()
                else:
                    alpha = 1.
                    alpha_loss = 0

                # Training Q Function
                target_q_min = torch.min(
                    target_soft_q_net1(next_state, new_next_action),
                    target_soft_q_net2(next_state, new_next_action)) - alpha * next_log_prob
                target_q_value = reward + (1 - done) * gamma * target_q_min
                q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
                q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

                soft_q_optimizer1.zero_grad()
                q_value_loss1.backward()
                soft_q_optimizer1.step()
                soft_q_optimizer2.zero_grad()
                q_value_loss2.backward()
                soft_q_optimizer2.step()

                # Training Policy Function
                predicted_new_q_value = torch.min(soft_q_net1(state, new_action),
                                                  soft_q_net2(state, new_action))
                policy_loss = (alpha * log_prob - predicted_new_q_value).mean()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Soft update the target value net
                for target_param, param in zip(target_soft_q_net1.parameters(),
                                               soft_q_net1.parameters()):
                    target_param.data.copy_(  # copy data value into target parameters
                        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                    )
                for target_param, param in zip(target_soft_q_net2.parameters(),
                                               soft_q_net2.parameters()):
                    target_param.data.copy_(  # copy data value into target parameters
                        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                    )

        if eps % save_interval:
            torch.save(soft_q_net1.state_dict(), model_path + '_q1')
            torch.save(soft_q_net2.state_dict(), model_path + '_q2')
            torch.save(policy_net.state_dict(), model_path + '_policy')

        
        for i, d in enumerate(info):
            if d.get('episode'):
                episode_reward = d['episode']['r']
                episode_length = d['episode']['l']
                print('worker {} episode reward {} episode length {}'
                      .format(i, episode_reward, episode_length))
                if len(rewards) == 0:
                    rewards.append(episode_reward)
                else:
                    rewards.append(rewards[-1] * 0.9 + episode_reward * 0.1)
                if len(rewards) % num_workers == 0:
                    clear_output(True)
                    plt.figure(figsize=(20, 5))
                    plt.plot(rewards)
                    plt.savefig('sac_v2_multi.png')
                    plt.clf()

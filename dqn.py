
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gym

from torch.utils.tensorboard import SummaryWriter
import datetime

current_time = datetime.datetime.now()
time_str = current_time.strftime("%y%m%d-%H%M")
log_dir_str = './log_dir/' + "dqn" + time_str
writer = SummaryWriter(log_dir=log_dir_str) 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ReplayBuffer(object):
    def __init__(self, size):
        self.buffer_size = size
        self.ptr = 0
        self.buffer = []
    
    def add(self, *transitions):
        # when the buffer is fully filled
        if len(self.buffer) == self.buffer_size:
            self.buffer[self.ptr] = transitions
            self.ptr += 1
        else:
            self.buffer.append(transitions)
            self.ptr += 1
        
        if self.ptr == self.buffer_size:
            self.ptr = 0 
   
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size=256):
        if self.__len__() < self.buffer_size:
            return None
        else:
            idxs = np.random.randint(0, self.buffer_size, size=batch_size)
            return [self.buffer[i] for i in idxs] 


class DQN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=16):
        super(DQN, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
         
    def forward(self, x):
        if isinstance(x, int):
           x = torch.tensor([x]).to(device)
        x = F.one_hot(x, num_classes=16).view(-1, 16).to(device).float()
        out = self.nn(x)
        return out
  
def select_action(obs, eps, net, env, greedy=False):
    if not greedy:
        e = np.random.rand()
        if eps > e:
            action = env.action_space.sample()
            return action
        
    out = net(obs)
    action = torch.argmax(out).item()
    return action

def update_parameters(transitions, net, optim, loss_func, i_step):
    if transitions is None:
        return
            
    buf_n_state = []
    buf_action = []
    buf_state = []
    buf_not_done = []
    buf_reward = []
    for t in transitions:
        state, action, n_state, reward, not_done = t
        buf_n_state.append(n_state)
        buf_action.append(action)
        buf_state.append(state)
        buf_not_done.append(float(not_done))
        buf_reward.append(reward)
   
    buf_n_state = torch.tensor(buf_n_state).view(-1,1).to(device) 
    buf_action = torch.tensor(buf_action).view(-1,1).to(device)
    buf_state = torch.tensor(buf_state).view(-1,1).to(device)
    buf_not_done = torch.FloatTensor(buf_not_done).view(-1, 1).to(device)
    buf_reward = torch.FloatTensor(buf_reward).view(-1,1).to(device)
    
    gamma = 0.99 
    td_target = buf_reward + buf_not_done * gamma * torch.max(net(buf_n_state.detach()), 1)[0].unsqueeze(1)
    
    loss = loss_func(td_target.detach(), torch.gather(net(buf_state), 1, buf_action))
    writer.add_scalar('Loss/train', loss, i_step)
    optim.zero_grad()
    loss.backward()
    optim.step()
   
def eval(env, net, i_step): 
    ep_rewards = []
    num_eval = 10
    eps = 0.0
    for idx in range(num_eval):
        obs = env.reset()
        done = False 
        ep_reward = 0.0
        while not done:
            action = select_action(obs, eps, net, env, greedy=True)
            next_obs, rew, done, info = env.step(action)
            ep_reward += rew
            obs = next_obs 
        ep_rewards.append(ep_reward) 
    
    avg_reward = sum(ep_rewards)/num_eval
    print("Episode reward = {}".format(avg_reward))
    writer.add_scalar('Reward/train', avg_reward, i_step)
 
def train(num_episode=100000):
    env = gym.make('FrozenLake-v1', is_slippery=True)
    net = DQN(env.observation_space.n, env.action_space.n).to(device)
    optimizer = Adam(net.parameters(), lr=3e-4)
    buf = ReplayBuffer(1000)
    eps_start = 0.9
    eps_stop = 0.1
    eps_decay = 0.99
    eval_period = 100
    training_start = 100
    loss_func = nn.MSELoss() 
    eps = eps_start
    i_step = 0
    for i_episode in range(num_episode):
        obs = env.reset()
        done = False
        ep_reward = 0.0 
        while not done:
            action = select_action(obs, eps, net, env)
            next_obs, rew, done, info = env.step(action)
            buf.add(obs, action, next_obs, rew, float(not done))
            ep_reward += rew 
            update_parameters(buf.sample(batch_size=100), net, optimizer, loss_func, i_step)
            i_step += 1 
            obs = next_obs
        
        if i_episode > training_start: 
            eps = max(eps * 0.99, eps_stop)
        if i_episode % eval_period == 0:
            eval(env, net, i_step)

    writer.flush()
    
if __name__ == "__main__":
    train()   
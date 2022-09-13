
'''
REINFORCE algorithm
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
import gym

from torch.utils.tensorboard import SummaryWriter
import datetime

current_time = datetime.datetime.now()
time_str = current_time.strftime("%y%m%d-%H%M")
log_dir_str = './log_dir/' + "pg" + time_str
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
    
    def clear(self):
        del self.buffer[:]
        self.buffer = []
        self.ptr = 0
   
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size=256):
        if self.__len__() < self.buffer_size:
            return None
        else:
            idxs = np.random.randint(0, self.buffer_size, size=batch_size)
            return [self.buffer[i] for i in idxs]
        
    def dump(self):
        return self.buffer[:self.ptr] 


class PolicyGradient(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=16):
        super(PolicyGradient, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax()
        )
         
    def forward(self, x):
        x = torch.tensor(x).to(device)
        # x = F.one_hot(x, num_classes=16).view(-1, 16).to(device).float()
        prob = self.nn(x)
        log_prob = torch.log(prob + 1e-8)
        return log_prob, prob
  
def select_action(obs, net, greedy=False):
    log_prob, prob = net(obs)
    if greedy:
        action = torch.argmax(prob).item()
    else:
        dist = Categorical(prob)
        action = dist.sample().item()
    return action

def update_parameters(transitions, net, optim, i_step):
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
   
    buf_n_state = torch.tensor(buf_n_state).view(-1,4).to(device) 
    buf_action = torch.tensor(buf_action).view(-1,1).to(device)
    buf_state = torch.tensor(buf_state).view(-1,4).to(device)
    buf_not_done = torch.FloatTensor(buf_not_done).view(-1, 1).to(device)
  
    gamma = 0.99
    buf_ret = []
   
    ret = 0.0 
    for reward in buf_reward[::-1]:
        ret = reward + gamma * ret
        buf_ret.append(ret) 
   
    buf_ret.reverse() 
    buf_ret = torch.FloatTensor(buf_ret).view(-1,1).to(device)
    
    log_prob, prob = net(buf_state)
    loss = -torch.sum(torch.gather(log_prob, 1, buf_action) * buf_ret.detach()) 
   
    writer.add_scalar('Loss/train', loss, i_step)
    optim.zero_grad()
    loss.backward()
    optim.step()
   
def eval(env, net, i_step): 
    ep_rewards = []
    num_eval = 10
    for idx in range(num_eval):
        obs = env.reset()
        done = False 
        ep_reward = 0.0
        while not done:
            action = select_action(obs, net, greedy=True)
            next_obs, rew, done, info = env.step(action)
            ep_reward += rew
            obs = next_obs
        ep_rewards.append(ep_reward) 
    
    avg_reward = sum(ep_rewards)/num_eval
    print("Episode reward = {}".format(avg_reward))
    writer.add_scalar('Reward/train', avg_reward, i_step)
 
def train(num_episode=100000):
    env = gym.make("CartPole-v1")
    net = PolicyGradient(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = Adam(net.parameters(), lr=3e-3)
    buf = ReplayBuffer(1000)
    eval_period = 100
    i_step = 0
    for i_episode in range(num_episode):
        obs = env.reset()
        done = False
        ep_reward = 0.0 
        while not done:
            action = select_action(obs, net)
            next_obs, rew, done, info = env.step(action)
            buf.add(obs, action, next_obs, rew, float(not done))
            ep_reward += rew 
            i_step += 1 
            obs = next_obs
        
        if ep_reward == 1:
            print("Get a reward at {}".format(i_episode))
            pass 
        update_parameters(buf.dump(), net, optimizer, i_step)
        buf.clear() 
        if i_episode % eval_period == 0:
            eval(env, net, i_step)

    writer.flush()
    
if __name__ == "__main__":
    train()   
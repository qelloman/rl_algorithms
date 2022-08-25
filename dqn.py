
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gym


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
        idxs = np.random.randint(0, self.buffer_size, size=batch_size)
        return self.buffer[idxs] 


class DQN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(out_dim, hidden_dim)
        )
         
    def forward(self, x):
        x = torch.tensor(x).view(-1,1).to(device)
        out = self.nn(x)
        return out
  
def select_action(obs, eps, net, env):
    e = np.random.rand()
    if eps > e:
        out = net(obs)
        action = torch.argmax(out).item()
    else:
        action = env.action_space.sample()
    return out

def update_parameters(transitions, net, optim):
    buf_n_state = []
    buf_action = []
    buf_state = []
    buf_done = []
    buf_reward = []
    for t in transitions:
        n_state, action, state, done, reward = t
        buf_n_state.append(n_state)
        buf_action.append(action)
        buf_state.append(state)
        buf_done.append(float(done))
        buf_reward.append(reward)
   
    buf_n_state = torch.tensor(buf_n_state).view(-1,1).to(device) 
    buf_action = torch.tensor(buf_action).view(-1,1).to(device)
    buf_state = torch.tensor(buf_state).view(-1,1).to(device)
    buf_done = torch.tensor(buf_done).view(-1, 1).to(device)
    buf_reward = torch.tensor(reward).to(device)
    
    gamma = 0.99 
    td_target = buf_reward + buf_done * gamma * torch.max(net(buf_n_state))
    loss = nn.MSELoss(td_target, net(buf_n_state).gather(buf_action, -1))
    optim.zero_grad()
    loss.backward()
    optim.step()
    
 
def train(num_episode=1000):
    env = gym.make('FrozenLake-v1')
    net = DQN(env.observation_space.n, env.action_space.n).to(device)
    optimizer = Adam(net.parameters(), lr=3e-3)
    buf = ReplayBuffer(10000)
    eps_start = 0.9
    eps_stop = 0.1
    eps_decay = 0.99 
    
    eps = eps_start 
    for i_episode in range(num_episode):
        obs = env.reset()
        done = False 
        while not done:
            action = select_action(obs, eps, net, env)
            next_obs, rew, done, info = env.step(action)
            buf.add(obs, action, next_obs, rew, done)
           
            update_parameters(buf.sample(batch_size=16), net, optimizer)
            
            obs = next_obs
             
        eps = np.max(eps * 0.99, eps_stop)

if __name__ == "__main__":
    train()   
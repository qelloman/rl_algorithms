
'''
soft actor-critic algorithm
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


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=16):
        super(Actor, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax()
        )
         
    def forward(self, state):
        state = torch.tensor(state).to(device)
        # x = F.one_hot(x, num_classes=16).view(-1, 16).to(device).float()
        prob = self.nn(state)
        log_prob = torch.log(prob + 1e-8)
        return log_prob, prob

class Critic(nn.Module): 
    def __init__(self, state_dim, action_dim, hidden_dim=16):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nn = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, state, action):
        action_onehot = F.one_hot(action, num_classes = self.action_dim)
        state_action = torch.concat([state, action_onehot], dtype=torch.float32)
        value = self.nn(state_action)
        return value
  
def select_action(obs, net, greedy=False):
    log_prob, prob = net(obs)
    if greedy:
        action = torch.argmax(prob).item()
    else:
        dist = Categorical(prob)
        action = dist.sample().item()
    return action

def update_parameters(transitions, actor, critic, target_critic, optim_actor, optim_critic, i_step):
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
    buf_reward = torch.FloatTensor(buf_reward).view(-1, 1).to(device)
    
    gamma = 0.99
    alpha = 0.1
    tau = 0.1
     
    buf_n_log_prob, buf_n_prob = actor(buf_n_state)
    n_dist = Categorical(buf_n_prob)
    buf_n_action = n_dist.sample()
    td_target = buf_reward + buf_not_done * gamma * (target_critic(buf_n_state, buf_n_action) - alpha * torch.gather(buf_n_log_prob, 1, buf_n_action)) 
    
    critic_loss = F.smooth_l1_loss(critic(buf_state), td_target.detach()) 
    writer.add_scalar('Critic Loss/train', critic_loss, i_step)
    optim_critic.zero_grad()
    critic_loss.backward()
    optim_critic.step()
   
    log_prob, prob = actor(buf_state)
    dist = Categorical(buf_n_prob)
    buf_action_sample = dist.sample()
    
    actor_loss = -torch.sum(alpha * torch.gather(log_prob, 1, buf_action_sample) - critic(buf_state, buf_action_sample)) 
    writer.add_scalar('Actor Loss/train', actor_loss, i_step)
    optim_actor.zero_grad()
    actor_loss.backward()
    optim_actor.step()
    
    for target_param, local_param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)  
   
def eval(env, actor, i_step): 
    ep_rewards = []
    num_eval = 10
    for idx in range(num_eval):
        obs = env.reset()
        done = False 
        ep_reward = 0.0
        while not done:
            action = select_action(obs, actor, greedy=True)
            next_obs, rew, done, info = env.step(action)
            ep_reward += rew
            obs = next_obs
        ep_rewards.append(ep_reward) 
    
    avg_reward = sum(ep_rewards)/num_eval
    print("Episode reward = {}".format(avg_reward))
    writer.add_scalar('Reward/train', avg_reward, i_step)
 
def train(num_episode=100000):
    env = gym.make("CartPole-v1")
    actor = Actor(env.observation_space.shape[0], env.action_space.n).to(device)
    critic = Critic(env.observation_space.shape[0], env.action_space.n).to(device)
    critic_target = critic.copy() 
    optim_actor = Adam(actor.parameters(), lr=3e-3)
    optim_critic = Adam(critic.parameters(), lr=3e-3)
    buf = ReplayBuffer(1000)
    batch_size = 100
    eval_period = 100
    i_step = 0
    for i_episode in range(num_episode):
        obs = env.reset()
        done = False
        ep_reward = 0.0 
        while not done:
            action = select_action(obs, actor)
            next_obs, rew, done, info = env.step(action)
            buf.add(obs, action, next_obs, rew, float(not done))
            ep_reward += rew 
            i_step += 1 
            obs = next_obs
            
            if len(buf) == batch_size:
                update_parameters(buf.dump(), actor, critic, critic_target, optim_actor, optim_critic, i_step)
                buf.clear() 
        
        if ep_reward == 1:
            print("Get a reward at {}".format(i_episode))
            pass
        if i_episode % eval_period == 0:
            eval(env, actor, i_step)

    writer.flush()
    
if __name__ == "__main__":
    train()   
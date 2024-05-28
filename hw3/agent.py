import gym
import pybullet_envs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

ENV = gym.make("InvertedPendulumSwingupBulletEnv-v0")
OBS_DIM = ENV.observation_space.shape[0]
ACT_DIM = ENV.action_space.shape[0]
ACT_LIMIT = ENV.action_space.high[0]
ENV.close()

#########################################################################################################################
############ 이 template에서는 DO NOT CHANGE 부분을 제외하고 마음대로 수정, 구현 하시면 됩니다                    ############
#########################################################################################################################

## 주의 : "InvertedPendulumSwingupBulletEnv-v0"은 continuious action space 입니다.
## Asynchronous Advantage Actor-Critic(A3C)를 참고하면 도움이 될 것 입니다.

class NstepBuffer:
    '''
    Save n-step trainsitions to buffer
    '''
    def __init__(self, n_step, gamma):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.n_step = n_step       
        self.gamma = gamma          

    def add(self, state, action, reward, next_state, done):
        '''
        add sample to the buffer
        '''
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self):
        '''
        sample transitions from buffer
        '''
        if len(self.states) < self.n_step:
            raise ValueError("Not enough elements in buffer to sample")

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in range(len(self.states) - self.n_step + 1):
            state, action = self.states[i], self.actions[i]
            reward, next_state, done = 0, self.next_states[i + self.n_step - 1], self.dones[i + self.n_step - 1]

            for j in range(self.n_step):
                reward += (self.gamma ** j) * self.rewards[i + j]
                if self.dones[i + j]:
                    done = True
                    break

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32)
        )
    
    def reset(self):
        '''
        reset buffer
        '''
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []


class ActorCritic(nn.Module):
    '''
    Pytorch module for Actor-Critic network
    '''
    def __init__(self):
        '''
        Define your architecture here
        '''
        super(ActorCritic, self).__init__()
        self.input_dim = 5 # State dimension
        self.hidden_dim = 256
        self.output_dim = 1 # Action dimension
        self.epsilon = 0.5
        self.train_step = 0
        self.epsilon_decay_interval = 1000

        # Actor network
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.log_std = nn.Parameter(torch.zeros(self.output_dim))  # Learnable parameter for std

        # Critic network
        self.fc3 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_layer = nn.Linear(self.hidden_dim, 1)  # Output a single value

        # Activation function
        self.activation = nn.Tanh()

    def actor(self, states):
        '''
        Get action distribution (mean, std) for given states
        '''
        x = self.activation(self.fc1(states))
        x = self.activation(self.fc2(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)  # Ensure std is positive
        return mean, std
        '''
        self.train_step += 1
        if self.train_step % self.epsilon_decay_interval == 0:
            if self.epsilon > 0.001:
                self.epsilon -= 0.001

        return action_mean, torch.Tensor([self.epsilon])
        '''

    def critic(self, states):
        '''
        Get values for given states
        '''
        x = self.activation(self.fc3(states))
        x = self.activation(self.fc4(x))
        value = self.value_layer(x)
        return value

class Worker(object):
    def __init__(self, global_actor, global_epi, sync, finish, n_step, seed):
        self.env = gym.make('InvertedPendulumSwingupBulletEnv-v0')
        self.env.seed(seed)
        self.lr = 0.0003
        self.gamma = 0.95
        self.entropy_coef = 0.005

        ############################################## DO NOT CHANGE ##############################################
        self.global_actor = global_actor
        self.global_epi = global_epi
        self.sync = sync
        self.finish = finish
        self.optimizer = optim.Adam(self.global_actor.parameters(), lr=self.lr)
        ###########################################################################################################  
        
        self.n_step = n_step
        self.local_actor = ActorCritic()
        self.nstep_buffer = NstepBuffer(n_step, self.gamma)

    def select_action(self, state):
        '''
        selects action given state

        return:
            continuous action value
        '''
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mean, std = self.local_actor.actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.clamp(-1.0, 1.0).detach().numpy()
    
    def train_network(self, states, actions, rewards, next_states, dones):
        '''
        Advantage Actor-Critic training algorithm
        '''
        # Convert data to tensors
        states = states.clone().detach()
        actions = actions.clone().detach()
        rewards = rewards.clone().detach()
        next_states = next_states.clone().detach()
        dones = dones.clone().detach()

        # Compute state values and advantages
        values = self.local_actor.critic(states).squeeze()
        next_values = self.local_actor.critic(next_states).squeeze()
        td_target = rewards + self.gamma * next_values * (1 - dones)
        advantages = td_target - values

        # Compute actor loss
        mean, std = self.local_actor.actor(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Compute critic loss
        critic_loss = F.mse_loss(values, td_target.detach().squeeze())

        # Compute entropy (for exploration)
        entropy = dist.entropy().mean()

        # Total loss
        total_loss = actor_loss + critic_loss - self.entropy_coef * entropy


        ############################################## DO NOT CHANGE ##############################################
        # Global optimizer update 준비
        self.optimizer.zero_grad()
        total_loss.backward()

        # Local parameter를 global parameter로 전달
        for global_param, local_param in zip(self.global_actor.parameters(), self.local_actor.parameters()):
                global_param._grad = local_param.grad

        # Global optimizer update
        self.optimizer.step()

        # Global parameter를 local parameter로 전달
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        ###########################################################################################################  

    def train(self):
        step = 1

        while True:
            state = self.env.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.nstep_buffer.add(state, action.item(), reward, next_state, done)

                # n step마다 한 번씩 train_network 함수 실행
                if step % self.n_step == 0 or done:
                    self.train_network(*self.nstep_buffer.sample())
                    self.nstep_buffer.reset()                    
                
                state = next_state
                step += 1
        
                if step == 30:
                    self.lr = 0.0005
                    self.entropy_coef = 0.01

                
            

            ############################################## DO NOT CHANGE ##############################################
            # 에피소드 카운트 1 증가                
            with self.global_epi.get_lock():
                self.global_epi.value += 1
            
            # evaluation 종료 조건 달성 시 local process 종료
            if self.finish.value == 1:
                break

            # 매 에피소드마다 global actor의 evaluation이 끝날 때까지 대기 (evaluation 도중 파라미터 변화 방지)
            with self.sync:
                self.sync.wait()
            ###########################################################################################################

        self.env.close()

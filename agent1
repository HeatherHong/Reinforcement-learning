import gym
import pybullet_envs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np          # added

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
    def __init__(self, n_step, gamma):      # modified
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.n_step = n_step        # added 
        self.gamma = gamma          # added


    def add(self, state, action, reward, next_state, done):
        '''
        add sample to the buffer
        '''
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)


    def sample(self):               # modified
        '''
        sample transitions from buffer
        '''
        return self.states, self.actions, self.rewards, self.next_states, self.dones

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

        self.input_size = 5
        self.hidden_size = 128
        self.sigma = 0.1
        self.counter = 0
        self.sigma_decrement_period = 3000

        self.actor_hidden = nn.Linear(OBS_DIM, self.hidden_size)
        self.actor_mu = nn.Linear(self.hidden_size, 1)

        self.critic_hidden = nn.Linear(OBS_DIM, self.hidden_size)
        self.critic_value = nn.Linear(self.hidden_size, 1)

    def actor(self, states):
        '''
        Get action distribution (mean, std) for given states
        '''
        x = torch.tanh(self.actor_hidden(states))
        mu = self.actor_mu(x)

        self.counter += 1
        if self.counter % self.sigma_decrement_period == 0:
            if self.sigma < 0.3:
                self.sigma += 0.01
                

        # noise = torch.randn_like(action_mean) * self.sigma
        # action_mean += noise
        # print(action_mean)

        return mu, torch.Tensor([self.sigma])


    def critic(self, states):
        '''
        Get values for given states
        '''
        x = torch.tanh(self.critic_hidden(states))
        value = self.critic_value(x)
        return value


class Worker(object):
    def __init__(self, global_actor, global_epi, sync, finish, n_step, seed):
        self.env = gym.make('InvertedPendulumSwingupBulletEnv-v0')
        self.env.seed(seed)
        self.lr = 0.000002
        self.gamma = 0.95
        self.entropy_coef = 0.007

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
        state = torch.FloatTensor(state).unsqueeze(0)
        mu,sigma = self.local_actor.actor(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action = action.clamp(-ACT_LIMIT, ACT_LIMIT)
        
        return action.detach().numpy()
    
    def train_network(self, states, actions, rewards, next_states, dones):
        '''
        Advantage Actor-Critic training algorithm
        '''
        values = self.local_actor.critic(torch.Tensor([states]))
        
        next_values = self.local_actor.critic(torch.Tensor([next_states]))
        # next_values[dones] = 0
        # print(rewards)
        targets = torch.Tensor(rewards) + self.gamma * next_values
        advantages = targets - values
        

        mu, std = self.local_actor.actor(torch.Tensor([states]))
        dist = Normal(mu, std)
        log_probs = dist.log_prob(torch.Tensor([actions]))
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy


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
                # print(action)
                next_state, reward, done, _ = self.env.step(action)
                self.nstep_buffer.add(state, action.item(), reward, next_state, done)

                # n step마다 한 번씩 train_network 함수 실행
                if step % self.n_step == 0 or done:
                    self.train_network(*self.nstep_buffer.sample())
                    self.nstep_buffer.reset()               
                
                state = next_state
                step += 1
            
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

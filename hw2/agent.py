import numpy as np

class Agent:

    def __init__(self, Q, mode):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        if self.mode == "test_mode":
                self.eps = 0
        if self.mode == "mc_control": # avg 6.53
                self.episode = []  # To store episodes for MC-control
                self.N = dict()            
                self.n_states = 500
                for S in range(self.n_states):
                    self.Q[S]
                self.eps = 1.0 
                self.gamma = 0.9
                self.alpha = 0.8
                self.episode_count = 1  # 用于追踪迭代次数，用于 ε 衰减 
        elif self.mode == "q_learning":# avg: 8.49
                self.eps = 1.0 
                self.gamma = 0.85
                self.alpha = 0.1



    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.random() < self.eps:  # Exploratory action
            action = np.random.choice(self.n_actions)
        else:  # Exploitative action
            action = np.argmax(self.Q[state])
        return action
    

    def step(self, state, action, reward, next_state, done):
        
        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # epsilon decay
        if done:
            if self.mode == "mc_control":
                # 如果回合结束，则更新回合数并按照 ε-greedy 策略衰减
                self.eps = 1/((self.episode_count//100) + 1)
                self.episode_count += 1
              
            elif self.mode == "q_learning":
                if self.eps > 0.01:
                    self.eps -= 0.00002

        if self.mode == "q_learning":
            # Update using the Q-learning formula
            best_next_action = max(self.Q[next_state])
            td_target = reward + self.gamma *  best_next_action
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error

        elif self.mode == "mc_control":
            # Store transitions for the entire episode
            self.episode.append((state, action, reward))

            if done:
              self.returns_sum = [[0 for j in range(self.n_actions)] for i in range(self.n_states)]

              for i in range(len(self.episode)-1, -1, -1):
                (S, A,  R) = self.episode[i]
                # Check if the (state, action) is first occurrence in this episode
                if (S, A) not in self.N.keys():
                  self.N[(S, A)] = 1
                else:
                  self.N[(S, A)] += 1
                self.returns_sum[S][A] += R + self.gamma * self.returns_sum[S][A]
                self.Q[S][A] += self.alpha * (self.returns_sum[S][A]/self.N[(S, A)] - self.Q[S][A])

              self.episode = []  # To store episodes for MC-control 
              
              
            


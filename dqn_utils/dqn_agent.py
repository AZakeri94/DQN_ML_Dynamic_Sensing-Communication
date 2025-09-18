import numpy as np
import torch as T
from dqn_utils.dqn_model import dqn_model
from dqn_utils.replay_memory import ReplayBuffer
import time 


class DQNAgent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, device,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='/models'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.device = device

        # Replay buffer
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        # Networks
        self.q_eval = dqn_model(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_'+self.algo+'_q_eval',
                                chkpt_dir=self.chkpt_dir,
                                device=self.device).to(self.device)

        self.q_next = dqn_model(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_'+self.algo+'_q_next',
                                chkpt_dir=self.chkpt_dir,
                                device=self.device).to(self.device)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float32, device=self.device).unsqueeze(0)
            actions = self.q_eval(state)  # shape [1, n_actions]
            action = T.argmax(actions[0]).item()  # pick first row
        else:
            action = np.random.choice(self.action_space)
        return action


    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        # Convert all at once, directly on device
        states, actions, rewards, states_, dones = map(
            lambda x: T.tensor(x, dtype=T.float32, device=self.device),
            (state, action, reward, new_state, done)
        )
        return states, actions.long(), rewards, states_, dones.bool()

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def save_models(self):
        print("... saving models ...")
        T.save({
            'q_eval_state_dict': self.q_eval.state_dict(),
            'q_next_state_dict': self.q_next.state_dict(),
            'optimizer_state_dict': self.q_eval.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter
        }, self.chkpt_dir + '/dqn_checkpoint.pth')

    def load_models(self):
        print("... loading models ...")
        checkpoint = T.load(self.chkpt_dir + '/dqn_checkpoint.pth', map_location=self.device)
        self.q_eval.load_state_dict(checkpoint['q_eval_state_dict'])
        self.q_next.load_state_dict(checkpoint['q_next_state_dict'])
        self.q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step_counter = checkpoint['learn_step_counter']

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        #t0 = time.time()    

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        #t1 = time.time()
        states, actions, rewards, states_, dones = self.sample_memory()
        #t2 = time.time()


        # Get predicted Q-values
        q_pred_all = self.q_eval(states)
        q_pred = q_pred_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get next Q-values
        q_next = self.q_next(states_).max(dim=1)[0].detach() # for img and pos information 
        
        q_next[dones] = 0.0

        # Compute target
        q_target = rewards + self.gamma * q_next

        # Loss and backward
        loss = self.q_eval.loss(q_pred, q_target)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()

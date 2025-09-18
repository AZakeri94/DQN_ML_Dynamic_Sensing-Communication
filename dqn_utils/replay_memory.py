import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size, state_dim, device):
        self.mem_size = max_size
        self.mem_cntr = 0
        #print(state_dim)
        self.state_dim = state_dim
        self.device = device

        # Preallocate memory for states, new states, actions, rewards, done flags
        self.state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        # Convert tensors to numpy
        self.state_memory[index] = state.cpu().numpy()
        self.new_state_memory[index] = state_[0].cpu().numpy()
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = torch.tensor(self.state_memory[batch], dtype=torch.float32)#.to(self.device) # NOTE: comment it for cpu, uncomment it for gpu
        states_ = torch.tensor(self.new_state_memory[batch], dtype=torch.float32)#.to(self.device)
        actions = torch.tensor(self.action_memory[batch])#.to(self.device)
        rewards = torch.tensor(self.reward_memory[batch])#.to(self.device)
        dones = torch.tensor(self.terminal_memory[batch])#.to(self.device)

        if self.device == 'cuda': # check before moving to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            states_ = states_.to(self.device)
            dones = dones.to(self.device)


        return states, actions, rewards, states_, dones

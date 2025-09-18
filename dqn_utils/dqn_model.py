import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class dqn_model(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, device=None):
        super(dqn_model, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = device if device is not None else T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)  # move model to the device
        #print(device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)  # Do not use ReLU on final Q-values
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
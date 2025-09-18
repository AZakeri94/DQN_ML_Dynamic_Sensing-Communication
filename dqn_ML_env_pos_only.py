import math
import random
import numpy as np
import torch as T
import torch.nn as nn
import pandas as pd
import ast
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from dqn_utils.ML_module_pos import NN_beam_pred
from torchvision.models import resnet18, ResNet18_Weights
import time
import torch.optim as optimizer

class dqn_ml_env_func:

    def __init__(self, V, sen_limit, data_len, csv_file, device=None):
        self.V = V
        self.sen_limit = sen_limit
        self.data_len = data_len
        self.csv_file = csv_file
        self.device = device if device is not None else T.device("cpu")


        # Load CSV once
        self.df = pd.read_csv(self.csv_file)

        #---
        self.num_classes = 33  
        checkpoint_path = './saved_folder/best_model/checkpoint/2-layer_nn_beam_pred'
        self.net = NN_beam_pred(2, self.num_classes).to(self.device)
        self.net.load_state_dict(T.load(checkpoint_path))
        self.net.eval()   

        self.opt = optimizer.Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-4
    )

        for p in self.net.parameters():
            p.requires_grad = False


    # %% getting the right ouput according to the sampling action 
    def get_sample(self, current_time):
        """Return full sample: image, GPS position, and label"""
        row = self.df.iloc[current_time]

        pos = T.tensor(ast.literal_eval(row["unit2_pos"]),
                    dtype=T.float32, device=self.device).unsqueeze(0)
        label = T.tensor(row["unit1_beam_32"],
                        dtype=T.long, device=self.device).unsqueeze(0)

        return pos, label

    def get_label(self, current_time):
        """Return only label"""
        row = self.df.iloc[current_time]
        return T.tensor(row["unit1_beam_32"], dtype=T.long,
                        device=self.device).unsqueeze(0)

    def get_pos(self, current_time):
        """Return only GPS position"""
        row = self.df.iloc[current_time]
        return T.tensor(ast.literal_eval(row["unit2_pos"]),
                        dtype=T.float32, device=self.device).unsqueeze(0)


    # %% Reset state every epoch
    def reset(self, n_iter, data_len, Nxt_Q, Nxt_age):
        rand_num = random.randint(0, self.data_len - n_iter) # this must be -n_iter, otherwise it might exceed the data length in the step function 

        Q_tensor = T.tensor([[float(Nxt_Q)]], dtype=T.float32, device=self.device)  # shape [1,1]
        pos_tensor = self.get_pos(rand_num)  # img: [1,3,224,224], pos: [1,2]
        age =  T.tensor([[float(Nxt_age)]], dtype=T.float32, device=self.device)

        # 5) Ensure pos is 2D already; if not, fix once
        if pos_tensor.dim() == 1:
            pos_tensor = pos_tensor.unsqueeze(0)          # [1,2]

        # 6) Concatenate in one go (all are already on the same device & dtype = float32)
        flat_state = T.cat((age, Q_tensor, pos_tensor), dim=1)  
        flat_state = flat_state.squeeze(0)         

        return flat_state, rand_num

    # %% Step
    def step(self, n_steps, rand_num, action, state):
        # state is composed as: [age | Q | pos]
        flat_state = state.view(-1) # torch.Size([515])
        current_Q = flat_state[1].item() # the integer value of virtual queu
        Nxt_Q = max(current_Q - self.sen_limit + action, 0) # update the queue

        Nxt_age = max(flat_state[0].item() + 1, 1e3) 

        pos_tensor = flat_state[2:].view(1, -1)         # .shape = torch.Size([2])

        
        current_time = n_steps + rand_num
        # Decide inputs based on action
        gps_input = pos_tensor
    
        if action == 1:
            gps_input = self.get_pos(current_time).view(1, -1)
            Nxt_age = 1 # reset age 

        
        # Move tensors to device for model forward
        gps_input_dev = gps_input.to(self.device).float()
        label = self.get_label(current_time)
        label_dev = label.to(self.device)
        #print("label", label_dev)

        with T.no_grad(): # runtime: 0.0003 sec
            self.opt.zero_grad()
            output = self.net( gps_input_dev ) # torch.Size([1, 33])
            #pred = T.argmax(output, dim=1).item()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, label_dev)

        # Reward
        q_reward = 0.5 * (Nxt_Q**2 - current_Q**2)
        #q_reward = current_Q * action
        reward = - self.V * loss.item() - q_reward

        #print("next q", Nxt_Q)

        Nxt_Q_tns = T.tensor([Nxt_Q])
        Nxt_age = T.tensor([Nxt_age])
        #print(Nxt_Q_tns)
        # Ensure 2D for concatenation
        if Nxt_Q_tns.dim() == 1:
            Nxt_Q_tns = Nxt_Q_tns.unsqueeze(0)  # [1, 1]
        if Nxt_age.dim() == 1:
            Nxt_age = Nxt_age.unsqueeze(0)    


        if gps_input.dim() == 1:
            gps_input = gps_input.unsqueeze(0)  # [1, 2]
        observation_ = T.cat((Nxt_age, Nxt_Q_tns, gps_input), dim=1)  # [1, 515]
 
        #print(observation_.shape)          

        return observation_, reward, Nxt_Q, output, label, Nxt_age

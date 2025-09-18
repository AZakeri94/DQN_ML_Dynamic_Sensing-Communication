import torch as T
import numpy as np
import random
import scipy.io  # primairly to read matdata
import os
import matplotlib.pyplot as plt

#from scipy.signal import pow2db
import time
startTime = time.time()

# import scripts
from dqn_utils.dqn_agent import DQNAgent 

#---
import warnings
warnings.filterwarnings("ignore")

# device 
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
import cProfile, pstats

from dqn_utils.topk_accuFrom_output import topK_accu

######################### System Parameter Con.  ##########################

sen_limit = 0.7
V = 1e3
data_len = 2300

action_mat = [ 0, 1 ]  # Indexed from 0-8, e.g., a[0] = [0, 0] ==>  alpha=0, beta=0


# %% ######################## the Envirenment  ##########################
csv_file='./scenario5_Allpos_beam_test_dqn.csv'

from dqn_ML_env_pos_only import dqn_ml_env_func as myenv

env = myenv( V, sen_limit, data_len, csv_file, device=None)

best_score = np.inf
load_checkpoint = True # re=run all the cells to get the right results
print("training mode ...") if not load_checkpoint else print("inference mode ...")
n_epoch = 300
n_iter =  400 # per each epoch

if __name__ == '__main__':
    agent = DQNAgent(gamma = 0.99999, epsilon = 0.99, lr = 0.001,
                     input_dims =  1+1+2  , # 
                     n_actions = 2,           #env.action_space.n,
                     mem_size = 50000, eps_min = 0.01,
                     batch_size = 64, replace = 1000, eps_dec = 1e-4,
                     device=device,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='dqn_ml'
                     )

    if load_checkpoint:
        agent.load_models()
        observation, _ = env.reset(1, data_len, 0, 0)
        rand_num = 1

    #---
    indx = 0
    x_, Q_t, Q_avr_t, snr_t  = np.zeros( n_epoch * n_iter), np.zeros( n_epoch * n_iter), np.zeros( n_epoch * n_iter),  np.zeros( n_epoch * n_iter)
    x_avr_t = np.zeros( n_epoch * n_iter)
    scores = np.zeros( n_epoch )

    #---
    correct1 = correct2 = correct3 = 0
    #---
    num_ones = round(sen_limit * 120000)   # round instead of floor
    seq = np.array([1] * num_ones + [0] * (120000 - num_ones))
    np.random.shuffle(seq)
    #print(seq)

    # Learning process
    Nxt_Q, Nxt_age = 0, 1
    for i in range(n_epoch):
        
        n_steps = 0; done = False
    
       # if not load_checkpoint: observation, rand_num = env.reset(n_iter, data_len, Nxt_Q, Nxt_age)
        observation, rand_num = env.reset(n_iter, data_len, Nxt_Q, Nxt_age) # this must be used if the inference is for number of times that is larger than the data length 
        
        while not done:
            
            # --- action selection ---  
            action = agent.choose_action(observation)
            #action = 1
            #action =   1 if (indx+1) % 2  == 0 else 0
            #action = 1 if random.random() < 0.2 else 0

            #--- random policy
            #action = seq[indx]

            observation_, reward, Nxt_Q, out, labels, Nxt_age = env.step(n_steps, rand_num,  action, observation) 

            if not load_checkpoint: # for training 
                    agent.store_transition(observation, action, reward, observation_, done)
                    agent.learn()
            observation = observation_
           
            # %%--- to compute top K accuracies
            #'''   
            label = int(labels) 
            if out.dim() == 2:
                out = out.squeeze(0)
            top1 = out.argmax().item()
            topk = out.topk(3).indices.tolist()
            correct1 += int(top1 == label)
            correct2 += int(label in topk[:2])
            correct3 += int(label in topk[:3])
           # ''' #--- 


            x_[indx] = action
            Q_t[indx] = Nxt_Q
            #snr_t[indx] = output
            if indx > 0:
               x_avr_t[indx] = np.mean(x_[:indx])  # Compute the mean if indx > 0
               Q_avr_t[indx] = np.mean(Q_t[:indx])
            else:
               x_avr_t[indx] = 0
               Q_avr_t[indx] = 0
            n_steps += 1
            indx += 1
            #print('----')
            if n_steps == n_iter: done = True

        print("epc#", i) if i % 50 == 0 else None

# to save the trained 
if not load_checkpoint:
    agent.save_models() 
    

filename = f"AvrgX_alpha{sen_limit}_V{V}_inference{load_checkpoint}.csv"
np.savetxt(filename, x_avr_t, delimiter=",")


print('Average num. sensing =', np.mean(x_))

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

# ---
top1_avg = correct1 / indx if indx else 0.0
top2_avg = correct2 / indx if indx else 0.0
top3_avg = correct3 / indx if indx else 0.0
print(f"Top-1 Accuracy: {top1_avg:.4f}")
print(f"Top-2 Accuracy: {top2_avg:.4f}")
print(f"Top-3 Accuracy: {top3_avg:.4f}")
print(f"avrg Accuracy: {(top1_avg+top2_avg+top3_avg)/3:.4f}")
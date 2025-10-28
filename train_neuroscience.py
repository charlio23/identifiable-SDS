import numpy as np
import os
import sys
from scipy.special import logsumexp, comb
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import argparse
import torch
from models.NeuralMSM import NeuralMSM

def parse_args():

    parser = argparse.ArgumentParser(description='Ar-HMM Data Gen and train')
    parser.add_argument('--exp_name', default='name', type=str, help='exp_name')
    parser.add_argument('--num_states', default=3, type=int, metavar='N', help='number of states')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use')
    parser.add_argument('--restarts_num', default=10, type=int, metavar='N', help='number of random restarts')
    parser.add_argument('--lag', default=2, type=int, metavar='N', help='number of lags')
    parser.add_argument('--causal', action='store_true', help='causal net')
    parser.add_argument('--activation', default='cos', type=str, help='activation')
    

    return parser.parse_args()

def train(num_lags, device, restarts_num):
    
    path = 'data/neuroscience/monkey_' + args.exp_name + '_train_visual_cortex.npy'
    obs = np.load(path)
    obs_test = torch.from_numpy(np.load('data/neuroscience/monkey_' + args.exp_name + '_test_visual_cortex.npy')).to(device).float()
    N, T, dim_obs = obs.shape

    learning_rates = [7e-3]
    hid_dims = [32, 128]
    gradient_clippings = [None]
    activations = [args.activation]
    
    best_model = NeuralMSM(num_states, dim_obs, hid_dim=1, device=device, lr=1, causal=args.causal, l1_penalty=0, l2_penalty=0, activation='cos', lag=num_lags)
    best_log_likeli = -np.inf
    batch_size = 210
    num_its = 2000
    for activation in activations:
        for hid_dim in hid_dims:
            for learning_rate in learning_rates:
                for gradient_clipping in gradient_clippings:
                    print("Activation:", activation, "Hid dim:", hid_dim, "Learning rate:", learning_rate, "Clipping:", gradient_clipping)
                    for i in range(restarts_num):
                        ## Random restarts
                        print("Restart", i)
                        model = NeuralMSM(num_states, dim_obs, hid_dim=hid_dim, device=device, lr=learning_rate, causal=False, 
                                    l1_penalty=0, l2_penalty=0, activation=activation, lag=num_lags, gradient_clipping=gradient_clipping)
                        log_likeli, _, _ = model.fit(torch.from_numpy(obs), num_its, batch_size=batch_size, early_stopping=20, max_scheduling_steps=2)
                        log_likeli = model.LogLikelihood(obs_test)
                        model.to('cpu')
                        #print(model.Q)
                        if best_log_likeli < log_likeli:
                            best_model = model
                            best_log_likeli = log_likeli
                            print("Best model is currently:", "Activation:", activation, "Hid dim:", hid_dim, "Learning rate:", learning_rate, "Clipping:", gradient_clipping, "LL:", best_log_likeli)
                        sys.stdout.flush()
    return best_model

args = parse_args()
num_states = args.num_states
n_lag = args.lag
device = args.device
restarts_num = args.restarts_num
best_model = train(n_lag, device, restarts_num)
print(best_model.Q)
sys.stdout.flush()
os.makedirs("results/neuroscience/".format(n_lag), exist_ok=True)
np.save("results/neuroscience/{}_num_states_{}_num_lags_{}_act_{}_exact_update.npy".format(args.exp_name, num_states, n_lag, args.activation), {'arr':best_model})
    
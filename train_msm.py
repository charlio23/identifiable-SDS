# Simple script to train a 2D MSM
import numpy as np
import torch

from models.NeuralMSM import NeuralMSM
from utils.benchmarks import benchmark_function_naive

# Settings (change on requirements - we assume a 2D MSM with 3 states is used)
device = 'cuda:1'
data_type = 'cosine'
data_size = 10000
T = 100
dim_obs = 2
num_states = 3
sparsity_prob = 0.0
seed = 23
# Load data
params = np.load('data/{}/params_N_{}_T_{}_dim_{}_state_{}_sparsity_{}_seed_{}.npy'.format(data_type,data_size,T, dim_obs, num_states, sparsity_prob, seed), allow_pickle=True).item()['arr']
obs = np.load('data/{}/observations_train_N_{}_T_{}_dim_{}_state_{}_sparsity_{}_seed_{}.npy'.format(data_type,data_size,T, dim_obs, num_states, sparsity_prob, seed))
# Train model (one restart - for optimal results using multiple restarts is advised)
# training time will show ~2 hous, but it will stop much earlier (~5 mins) due to early stopping.
model = NeuralMSM(num_states, dim_obs, hid_dim=8, device=device, lr=7e-3, causal=True, l1_penalty=0, l2_penalty=0, activation='cos')
log_likeli, _, _ = model.fit(torch.from_numpy(obs), 100, batch_size=100, early_stopping=4, max_scheduling_steps=2)
# For monitoring model.fit also outputs accuracy list if verbose_path is provided
print(model.Q)
print(benchmark_function_naive(model.transitions.cpu().float(), params, dim_obs, net=data_type))


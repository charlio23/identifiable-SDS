import argparse
import os
import sys

import numpy as np
from scipy.special import logsumexp, comb
import torch
from tqdm import tqdm


from models.PolyMSM import sieve as sieve_poly
from models.PolyMSM import calc_dist_params as calc_dist_params_poly
from models.NeuralMSM import NeuralMSM
from utils.benchmarks import benchmark_function_naive
from utils.transitions import (get_trans_mat, func_cosine_with_sparsity, func_polynomial, 
    func_softplus_with_sparsity, sample_adj_mat)

def parse_args():

    parser = argparse.ArgumentParser(description='Ar-HMM Data Gen and train')
    parser.add_argument('--seeds', default=[23], nargs="+", type=int, metavar='N', help='number of seeds (multiple seeds run multiple experiments)')
    parser.add_argument('--dim_obs', default=2, type=int, metavar='N', help='number of dimensions')
    parser.add_argument('--num_states', default=3, type=int, metavar='N', help='number of states')
    parser.add_argument('--sparsity_prob', default=0.0, type=float, metavar='N', help='sparsity probability')
    parser.add_argument('--data_type', default='cosine', type=str, help='Type of data generated (cosine|poly)')
    parser.add_argument('--generate_data', action='store_true', help='Generate data and then train')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use')
    parser.add_argument('--degree', default=3, type=int, metavar='N', help='degree of polynomial')
    parser.add_argument('--restarts_num', default=10, type=int, metavar='N', help='number of random restarts')
    parser.add_argument('--seq_length', default=200, type=int, metavar='N', help='number of timesteps')
    parser.add_argument('--num_samples', default=10000, type=int, metavar='N', help='number of samples')
    
    return parser.parse_args()


def generate_data(seed, num_states, dim_obs, T, data_size, sparsity_prob='0.0', data_type='cosine', degree=3, save=True):
    np.random.seed(seed)
    # Init distrib and transition matrix
    pi = np.array([1./num_states]*num_states)
    Q = get_trans_mat(num_states)

    # Transition parametrisations
    if data_type=='poly':
        num_params = int(comb(dim_obs+degree, degree))
        params = (np.random.rand(num_states, dim_obs, num_params)-0.5)
        params[:,:,1:] *= 0.05
    elif data_type=='softplus':
        params = [(np.random.randn(1, dim_obs,8)*0.25, np.random.randn(8, dim_obs,dim_obs)*0.25, np.random.randn(dim_obs, 8)*0.25, sample_adj_mat(sparsity_prob, dim_obs)) for k in range(num_states)]
    elif data_type=='cosine':
        params = [(np.random.randn(1, dim_obs,8)*0.5, np.random.randn(8, dim_obs,dim_obs)*0.5, np.random.randn(dim_obs, 8)*0.5, sample_adj_mat(sparsity_prob, dim_obs)) for k in range(num_states)]
    else:
        raise NotImplementedError('Data type: ' + data_type + ' is not implemented')
    scales = np.array([0.05]*num_states)

    # Initial mean params
    means = [np.random.randn(dim_obs)*.7 for i in range(num_states)]

    # Data generatio loop (could be made more efficient)
    for (N, mode) in zip([data_size, data_size//10], ['train', 'test']):
        state = np.random.choice(num_states,size=N, p=pi)
        obs = np.zeros((N,T,dim_obs))
        latent_states = np.zeros((N,T))
        # Generate data
        latent_states[:,0] = state
        obs[:,0,:] = np.array([np.random.normal(loc=means[state[n]], scale=0.1) for n in range(N)]).reshape(N,-1)
        for i in tqdm(range(1,T)):
            # Next state
            state = np.array([int(np.random.choice(num_states, p=Q[state[n],:])) for n in range(N)])
            latent_states[:,i] = state
            # Generate obs
            # Scales
            scale_ = scales[state]
            # Means
            if data_type=='poly':
                means_ = func_polynomial(obs[:,i-1,:], params[state], degree=degree)
            elif data_type=='cosine':
                means_ = [func_cosine_with_sparsity(obs[n,i-1,:], params[state[n]]) for n in range(N)]
            elif data_type=='softplus':
                means_ = [func_softplus_with_sparsity(obs[n,i-1,:], params[state[n]]) for n in range(N)]
            else:
                raise NotImplementedError('Data type: ' + data_type + ' is not implemented')
            obs[:,i,:] = np.array([np.random.multivariate_normal(means_[n], cov=scale_[n]*np.eye(dim_obs)) for n in range(N)])
        os.makedirs("data/{}".format(data_type), exist_ok=True)
        np.save('data/{}/observations_{}_N_{}_T_{}_dim_{}_state_{}_sparsity_{}_seed_{}.npy'.format(data_type,mode,data_size,T, dim_obs, num_states, sparsity_prob,seed), obs)
    np.save('data/{}/params_N_{}_T_{}_dim_{}_state_{}_sparsity_{}_seed_{}.npy'.format(data_type,data_size,T, dim_obs, num_states, sparsity_prob,seed), {'arr':params})

    return obs, params


def train(obs, params, device, restarts_num, data_type, degree=3):
    N, T, dim_obs = obs.shape
    if data_type=='poly':
        # Polynomial training
        if T==10:
            batch_size=1000
        else:
            batch_size = 1000
        best_model, log_likeli = sieve_poly(num_states, dim_obs, torch.from_numpy(obs), coefs=degree, device=args.device, batch_size=batch_size)
        dist, _, _ = calc_dist_params_poly(best_model.means.cpu().float(), params, dim_obs, mode='naive', method='grid', degree=degree)
    else:
        learning_rate = 7e-3
        hid_dim = 8
        activation = 'softplus'
        if args.data_type=='cosine':
            activation='cos'
        best_model = NeuralMSM(num_states, dim_obs, hid_dim=1, device=device, lr=1, causal=False, l1_penalty=0, l2_penalty=0, activation='cos')
        best_log_likeli = -np.inf

        batch_size = 1000
        num_its = 1000
        if T==10:
            batch_size=1000
            num_its = 2000
        #if dim_obs >= 10:
        #    batch_size = 64
        #    num_models = 3
        print("Activation:", activation, "Hid dim:", hid_dim, "Learning rate:", learning_rate)
        for i in range(restarts_num):
            ## Random restarts
            print("Restart", i)
            model = NeuralMSM(num_states, dim_obs, hid_dim=hid_dim, device=device, lr=learning_rate, causal=True, l1_penalty=0, l2_penalty=0, activation=activation)
            log_likeli, _, _ = model.fit(torch.from_numpy(obs), num_its, batch_size=batch_size, early_stopping=4, max_scheduling_steps=2)
            model.to('cpu')
            print(model.Q)
            dist, _ = benchmark_function_naive(model.transitions.cpu().float(), params, dim_obs, net=data_type)
            if best_log_likeli < log_likeli:
                best_model = model
                best_log_likeli = log_likeli
                print("Best model is currently:", "Activation:", activation, "Hid dim:", hid_dim, "Learning rate:", learning_rate, "LL:", best_log_likeli, "dist:", dist)
            sys.stdout.flush()
        
        dist, _ = benchmark_function_naive(best_model.transitions.cpu().float(), params, dim_obs, net=data_type)
    return best_model, dist

if __name__=="__main__":
    
    args = parse_args()
    data_type = args.data_type
    data_size = args.num_samples
    T = args.seq_length
    dim_obs = args.dim_obs
    num_states = args.num_states
    sparsity_prob = args.sparsity_prob
    device = args.device
    degree = args.degree
    restarts_num = args.restarts_num
    distances = np.zeros(len(args.seeds))
    for k, seed in enumerate(args.seeds):
        if args.generate_data:
            obs, params = generate_data(seed, num_states, dim_obs, T, data_size, sparsity_prob, data_type, degree, save=False)
        else:
            params = np.load('data/{}/params_N_{}_T_{}_dim_{}_state_{}_sparsity_{}_seed_{}.npy'.format(data_type,data_size,T, dim_obs, num_states, sparsity_prob, seed), allow_pickle=True).item()['arr']
            obs = np.load('data/{}/observations_train_N_{}_T_{}_dim_{}_state_{}_sparsity_{}_seed_{}.npy'.format(data_type,data_size,T, dim_obs, num_states, sparsity_prob, seed))
        best_model, dist = train(obs, params, device, restarts_num, data_type, degree)
        print(best_model.Q)
        print("Best model dist:", dist.item())
        sys.stdout.flush()
        os.makedirs("results/{}/".format(data_type), exist_ok=True)
        np.save("results/{}/inferred_params_N_{}_T_{}_dim_{}_state_{}_sparsity_{}_seed_{}.npy".format(data_type,data_size,T, dim_obs, num_states, sparsity_prob, seed), {'arr':best_model})
        distances[k] = dist.item()
        np.save("results/{}/distances_N_{}_T_{}_dim_{}_state_{}_sparsity_{}.npy".format(data_type,data_size,T, dim_obs, num_states, sparsity_prob), distances)
        
import argparse
import os
import sys
import time

import cv2
import numpy as np
from scipy.special import logsumexp, comb
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from dataloaders.BouncingBallDataLoader import BouncingBallDataLoader
from models.modules import MLP
from models.VariationalSNLDS import VariationalSNLDS
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from utils.transitions import (get_trans_mat, func_cosine_with_sparsity, func_polynomial, 
    func_leaky_relu, func_softplus_with_sparsity, sample_adj_mat)

def save_checkpoint(state, filename='model'):
    os.makedirs("results/models_sds/", exist_ok=True)
    torch.save(state, "results/models_sds/" + filename + '.pth.tar')

def parse_args():

    parser = argparse.ArgumentParser(description='Ar-HMM Data Gen and train')
    parser.add_argument('--seeds', default=[24], nargs="+", type=int, metavar='N', help='number of seeds (multiple seeds run multiple experiments)')
    parser.add_argument('--dim_obs', default=2, type=int, metavar='N', help='number of dimensions')
    parser.add_argument('--dim_latent', default=2, type=int, metavar='N', help='number of latent dimensions')
    parser.add_argument('--num_states', default=3, type=int, metavar='N', help='number of states')
    parser.add_argument('--sparsity_prob', default=0.0, type=float, metavar='N', help='sparsity probability')
    parser.add_argument('--data_type', default='cosine', type=str, help='Type of data generated (cosine|poly)')
    parser.add_argument('--generate_data', action='store_true', help='Generate data and then train')
    parser.add_argument('--no-train', action='store_true', help='Activate no train')
    parser.add_argument('--images', action='store_true', help='Use images')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use')
    parser.add_argument('--degree', default=3, type=int, metavar='N', help='degree of polynomial')
    parser.add_argument('--restarts_num', default=10, type=int, metavar='N', help='number of random restarts')
    parser.add_argument('--seq_length', default=200, type=int, metavar='N', help='number of timesteps')
    parser.add_argument('--num_samples', default=5000, type=int, metavar='N', help='number of samples')
    
    return parser.parse_args()

def _draw(points, res=32):
    """Returns array of the environment evolution

    Args:
        res (int): Image resolution (images are square).
        color (bool): True if RGB, false if grayscale.

    Returns:
        vid (np.ndarray): Rendered rollout as a sequence of images
    """
    max_x = 4
    max_y = 3
    min_x = -3
    min_y = -4
    T, _ = points.shape
    vid = np.zeros((T, res, res, 3), dtype='float')
    background_color = [81./255, 88./255, 93./255]
    ball_color = [173./255, 146./255, 0.]
    space_res = (max_x - min_x)/res
    for t in range(T):
        point = (int((points[t][0] - min_x)/space_res), int((points[t,1] - min_y)/space_res))
        vid[t] = cv2.circle(vid[t], point,
                            int(1/space_res), ball_color, -1)
        vid[t] = cv2.blur(cv2.blur(vid[t], (1, 1)), (2, 2))
    vid += background_color
    vid[vid > 1.] = 1.
    return vid

def generate_data(seed, num_states, dim_obs, dim_latents, T, data_size, sparsity_prob='0.0', data_type='cosine', degree=3, save=True, images=False):
    np.random.seed(seed)
    # Init distrib and transition matrix
    pi = np.array([1./num_states]*num_states)
    Q = get_trans_mat(num_states)

    if data_type=='poly':
        num_params = int(comb(dim_latents+degree, degree))
        params = (np.random.rand(num_states, dim_latents, num_params)-0.5)
        params[:,:,1:] *= 0.05
    elif data_type=='cosine':
        params = [(np.random.randn(1, dim_latents,8)*0.5, np.random.randn(8, dim_latents,dim_latents)*0.5, np.random.randn(dim_latents, 8)*0.5, sample_adj_mat(sparsity_prob, dim_latents)) for k in range(num_states)]
    else:
        params = [(np.random.randn(1, dim_latents,8)*0.25, np.random.randn(8, dim_latents,dim_latents)*0.25, np.random.randn(dim_latents, 8)*0.25, sample_adj_mat(sparsity_prob, dim_latents)) for k in range(num_states)]
    scales = np.array([0.05]*num_states)
    obs_distribs = [(param, scale) for param, scale in zip(params, scales)]
    means = [np.random.randn(dim_latents)*.7 for i in range(num_states)]
    params_leaky = (np.random.randn(dim_obs,8)*0.5, np.random.randn(8,dim_latents)*0.5, np.random.randn(8)*0.5)

    for (N, mode) in zip([data_size, data_size//10], ['train', 'test']):
        state = np.random.choice(num_states,size=N, p=pi)
        latents = np.zeros((N,T,dim_latents))
        obs = np.zeros((N,T,dim_obs))
        latent_states = np.zeros((N,T))
        # Generate data
        latent_states[:,0] = state
        latents[:,0,:] = np.array([np.random.normal(loc=means[state[n]], scale=0.1) for n in range(N)]).reshape(N,-1)
        obs[:,0,:] = func_leaky_relu(latents[:,0,:], params_leaky)
        for i in tqdm(range(1,T)):
            # Next state
            state = np.array([int(np.random.choice(num_states, p=Q[state[n],:])) for n in range(N)])
            latent_states[:,i] = state
            # Generate latents
            # Scales
            scale_ = scales[state]
            # Means
            if data_type=='poly':
                means_ = func_polynomial(latents[:,i-1,:], params[state], degree=degree)
            elif data_type=='cosine':
                means_ = [func_cosine_with_sparsity(latents[n,i-1,:], params[state[n]]) for n in range(N)]
            else:
                means_ = [func_softplus_with_sparsity(latents[n,i-1,:], params[state[n]]) for n in range(N)]
            latents[:,i,:] = np.array([np.random.multivariate_normal(means_[n], cov=scale_[n]*np.eye(dim_latents)) for n in range(N)])
            obs[:,i,:] = func_leaky_relu(latents[:,i,:], params_leaky)
        os.makedirs("data/latent_variables", exist_ok=True)
        np.save('data/latent_variables/latents_{}_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}.npy'.format(mode,N,T, dim_latents, dim_obs, num_states, sparsity_prob, data_type, seed), latents)
        np.save('data/latent_variables/states_{}_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}.npy'.format(mode,N,T, dim_latents, dim_obs, num_states, sparsity_prob, data_type, seed), latent_states)
        if not images:
            np.save('data/latent_variables/obs_{}_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}.npy'.format(mode,N,T, dim_latents, dim_obs, num_states, sparsity_prob, data_type, seed), obs)
        else:
            path = "data/latent_variables/images_{}_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}/".format(mode,N,T, dim_latents, dim_obs, num_states, sparsity_prob, data_type, seed)
            os.makedirs(path, exist_ok=True)
            for n in tqdm(range(N)):
                vid = _draw(latents[n],32)
                filename = "{0:05d}".format(n)
                np.savez(os.path.join(path, filename), vid)
    np.save('data/latent_variables/params_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}.npy'.format(N,T, dim_latents, dim_obs, num_states, sparsity_prob, data_type, seed), {'arr_latents':params, 'arr_leaky':params_leaky, 'init_means':means})


def train(path, num_states, dim_obs, dim_latent, T, data_size, sparsity_prob, data_type, device, seed):

    if args.images:
        dl = BouncingBallDataLoader(path)
        exp_name = 'inferred_params_images_N_{}_T_{}_dim_latent_{}_state_{}_sparsity_{}_net_{}_seed_{}'.format(data_size,
                T, dim_latent, num_states, sparsity_prob, data_type, seed)
    else:
        dl = TensorDataset(torch.from_numpy(np.load(path)))
        exp_name = 'inferred_params_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}'.format(data_size,
                T, dim_latent, dim_obs, num_states, sparsity_prob, data_type, seed)
    final_temperature = 1

    ## Hyperparameters (highly advisable to modify on data type and size)
    ## In this configuration all the models use the temperature scheduling indicated in Appendix F.3
    ## https://arxiv.org/abs/2305.15925
    ## If the models runs into state collapse, we advise increasing the temperature and decay rate.
    ## If the temperature is increased, it is advised to increase the scheduler_epochs to prevent fake convergence of Q
    if args.images:
        pre_train_check = 10
        init_temperature = 10
        iter_update_temp = 50
        iter_check_temp = 200
        epoch_num = 200
        learning_rate = 5e-4
        gamma_decay = 0.8
        scheduler_epochs = 80
        decay_rate = 0.975
    else:           
        pre_train_check = 5
        init_temperature = 5
        iter_update_temp = 50
        iter_check_temp = 1000
        epoch_num = 100
        learning_rate = 5e-4
        gamma_decay = 0.5
        scheduler_epochs = 40
        decay_rate = 0.9
    
    for restart_num in range(args.restarts_num):
        best_elbo = -torch.inf
        if args.images:
            dataloader = DataLoader(dl, batch_size=32, shuffle=True)
        else:
            dataloader = DataLoader(dl, batch_size=50, shuffle=True)
        model = VariationalSNLDS(dim_obs, dim_latent, 64, num_states, encoder_type='video' if images else 'recurent', device=device, annealing=False, inference='alpha', beta=0)
        # Useful for setting a smaller transition network to avoid overfitting
        model.transitions = torch.nn.ModuleList([MLP(dim_latent, dim_latent, 16, 'cos') for _ in range(num_states)]).to(device).float()

        model.temperature = init_temperature
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_epochs, gamma=gamma_decay)
        iterations = 0
        model.beta = 0
        mse = 1e5
        model.Q.requires_grad_(False)
        model.pi.requires_grad_(False)
        for epoch in range(0, epoch_num):
            if epoch >= pre_train_check and mse > 6e3:
                break
            if epoch >= pre_train_check and epoch < scheduler_epochs//4:
                model.beta = 1
                if args.images: # With images we will use high temperature annealing. No need for long warmups
                    model.Q.requires_grad_(True)
                    model.pi.requires_grad_(True)
            elif epoch >= scheduler_epochs//4:
                model.Q.requires_grad_(True)
                model.pi.requires_grad_(True)
            end = time.time()
            for i, (sample,) in enumerate(dataloader, 1):
                if args.images:
                    B, T, C, H, W = sample.size()
                else:
                    B, T, D = sample.size()
                obs_var = Variable(sample[:,:].float(), requires_grad=True).to(device)
                optimizer.zero_grad()
                x_hat, _, _, losses = model(obs_var)
                # Compute loss and optimize params
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if args.images:
                    mse = torch.nn.functional.mse_loss(x_hat.reshape(B, T, C, H, W), obs_var, reduction='sum')/(B)
                else:
                    mse = torch.nn.functional.mse_loss(x_hat, obs_var, reduction='sum')/(B)
                batch_time = time.time() - end
                end = time.time()   
                iterations +=1
                if iterations%iter_update_temp==0 and iterations >= iter_check_temp:
                    model.temperature = model.temperature*decay_rate
                    model.temperature = max(model.temperature, final_temperature)
                if i%5==0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time:.3f}\t'
                        'ELBO {loss:.4e}\t MSE: {mse:.4e}\t MSM: {msm:.4e}'.format(
                        epoch, i, len(dataloader), batch_time=batch_time, 
                        loss=losses['elbo'], mse=mse, msm=losses['msm_loss']))
                    sys.stdout.flush()

            if epoch%2==0:
                print((model.Q/model.temperature).softmax(-1))
                print((model.pi/model.temperature).softmax(-1))
                print(model.temperature)
            sys.stdout.flush()
            scheduler.step()
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict()
            }, filename=exp_name+ '_restart_{:02d}'.format(restart_num))
            if best_elbo < losses['elbo']:
                best_elbo = losses['elbo']
                save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict()
                }, filename=exp_name+'_best_model')

if __name__=="__main__":
    
    args = parse_args()
    data_type = args.data_type
    data_size = args.num_samples
    T = args.seq_length
    dim_obs = args.dim_obs
    dim_latent = args.dim_latent
    num_states = args.num_states
    sparsity_prob = args.sparsity_prob
    images = args.images
    device = args.device
    degree = args.degree
    restarts_num = args.restarts_num
    for k, seed in enumerate(args.seeds):
        if args.generate_data:
            generate_data(seed, num_states, dim_obs, dim_latent, T, data_size, sparsity_prob, data_type, degree, save=False, images=images)
        if not images:
            path = 'data/latent_variables/obs_train_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}.npy'.format(data_size,T, dim_latent, dim_obs, num_states, sparsity_prob, data_type, seed)
        else:
            path = "data/latent_variables/images_train_N_{}_T_{}_dim_latent_{}_dim_obs_{}_state_{}_sparsity_{}_net_{}_seed_{}/".format(data_size,T, dim_latent, dim_obs, num_states, sparsity_prob, data_type, seed)
        if not args.no_train:
            train(path, num_states, dim_obs, dim_latent, T, data_size, sparsity_prob, data_type, device, seed)
            print("Trained Seed:", seed)
        sys.stdout.flush()

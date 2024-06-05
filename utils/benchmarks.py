import itertools

import numpy as np
import torch
from torch import nn

def function_dist(input_func, target_func, grid, net='cosine'):
    inferred_response = input_func(grid)
    if net == 'cosine':
        if len(target_func)==3:
            target_response = func_cosine(grid, target_func)
        else:
            target_response = func_cosine_with_sparsity(grid, target_func)
    elif net=='network':
        target_response = target_func(grid)
    elif net=='softplus':
        target_response = func_softplus_with_sparsity(grid, target_func)
    else:
        raise NotImplementedError(net + " is not implemented")
    return nn.functional.mse_loss(inferred_response, target_response)/nn.functional.mse_loss(target_response,torch.zeros_like(target_response))

def create_grid(dim, num_points, left_lim, right_lim, MC=True):
    if MC:
        grid = np.array([[np.random.uniform(left_lim, right_lim) for _ in range(dim)] for _ in range(1000)])
    else:
        grid_edges = [np.linspace(left_lim, right_lim, num_points) for _ in range(dim)]
        grid = np.stack(np.meshgrid(*grid_edges))
        grid.reshape(dim, -1).T
    return grid

def func_cosine(x, features):
    alphas, omegas, betas = features
    result = torch.matmul(torch.from_numpy(omegas[None,:,:]).float(),x[:,:,None])[:,:,0]
    result = torch.cos(result + torch.from_numpy(betas[None,:]).float())
    result = torch.matmul(torch.from_numpy(alphas[None,:,:]).float(),result[:,:,None])[:,:,0]
    return result

def func_cosine_with_sparsity(x, features):
    alphas, omegas, betas, adj_mat = features
    dim_obs = adj_mat.shape[0]
    N, _ = x.shape
    out = torch.zeros(N,dim_obs)
    for i in range(dim_obs):
        input = x*torch.from_numpy(adj_mat[None,i,:]).float()
        result = torch.matmul(torch.from_numpy(omegas[None,:,i,:]).float(),input[:,:,None])[:,:,0]
        result = torch.cos(result + torch.from_numpy(betas[None,i]).float())
        out[:,i] = torch.matmul(torch.from_numpy(alphas[None,:,i,:]).float(),result[:,:,None])[:,0,0]
    return out


def func_softplus_with_sparsity(x, features):
    alphas, omegas, betas, adj_mat = features
    dim_obs = adj_mat.shape[0]
    N, _ = x.shape
    out = torch.zeros(N,dim_obs)
    for i in range(dim_obs):
        input = x*torch.from_numpy(adj_mat[None,i,:]).float()
        result = torch.matmul(torch.from_numpy(omegas[None,:,i,:]).float(),input[:,:,None])[:,:,0]
        result = torch.nn.functional.softplus(result + torch.from_numpy(betas[None,i]).float())
        out[:,i] = torch.matmul(torch.from_numpy(alphas[None,:,i,:]).float(),result[:,:,None])[:,0,0]
    return out

def benchmark_function_all(inferred_funcs, gt_funcs, dim, net='cosine'):
    grid = torch.from_numpy(create_grid(dim, 10000, -1, 1)).float()
    K = len(inferred_funcs)
    distances = []
    permutations = np.array(list(itertools.permutations(range(K))))
    for perm in permutations:
        distance = torch.tensor([function_dist(inferred_funcs[p], gt_funcs[idx], grid, net=net) for idx,p in enumerate(perm)]).sum()/K
        distances.append(distance)

    distances = np.array(distances)
    idx = np.argmin(distances)
    dist = distances[idx]
    perm = permutations[idx]
    return (dist, perm)

def benchmark_function_naive(inferred_funcs, gt_funcs, dim, net='cosine'):
    # Assuming all networks are approximated well, we can iterate to find the minimum distance
    # for each mode (search is O(K^2))
    grid = torch.from_numpy(create_grid(dim, 10000, -1, 1)).float()
    K = len(inferred_funcs)
    idxs = list(range(K))
    permutation = []
    current_net = 0
    dist = 0
    while(len(idxs) != 0):
        min_dist = torch.inf
        current_gt_mode = gt_funcs[current_net]
        idx_max = idxs[0]
        for i in idxs:
            act_dist = function_dist(inferred_funcs[i], current_gt_mode, grid, net=net)
            if (min_dist > act_dist):
                min_dist = act_dist
                idx_max = i
        current_net += 1
        permutation.append(idx_max)
        dist += min_dist
        idxs.remove(idx_max)
    dist /= K

    return dist, permutation

def benchmark_function(inferred_funcs, gt_funcs, dim, net='cosine', mode='all'):
    if mode=='all':
        return benchmark_function_all(inferred_funcs, gt_funcs, dim, net)
    return benchmark_function_naive(inferred_funcs, gt_funcs, dim, net)
    

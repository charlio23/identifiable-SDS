import itertools

import numpy as np
import torch
from scipy.stats import multivariate_normal
from scipy.special import logsumexp, comb
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from models.MSM import MSM

class PolyMSM(MSM):

    def __init__(self, num_states, obs_dim, coefs=None, device='cpu'):
        
        super().__init__(num_states, obs_dim, device)
        self.coefs = coefs

        # Initialise init mean and covariance 
        self.init_mean = (torch.rand(self.num_states, self.obs_dim)-0.5).to(device).float()
        self.init_cov = (torch.rand(self.num_states,1,1)*torch.eye(self.obs_dim)[None,:,:]).to(device).float()

        num_params = int(comb(self.obs_dim + self.coefs, self.coefs))
        self.covs = (torch.eye(self.obs_dim)[None,:,:]).to(device).float()*5
        self.means = (torch.rand(self.num_states, self.obs_dim, num_params)-0.5).to(device).float()
        self.means[:,:,1:] *= 0
   
    def _compute_local_evidence(self ,obs, obs_one):
        N, T, _ = obs.shape
        init_distrib_ = torch.distributions.MultivariateNormal(self.init_mean, self.init_cov)
        log_local_evidence_1 = init_distrib_.log_prob(obs[:,0:1,None,:].repeat(1,1,self.num_states,1))
        means_ = torch.cat([torch.matmul(self.means[None,None,None,i,:,:],obs_one[:,:,None,:,None])[:,:,:,:,0] for i in range(self.num_states)], dim=2)
        covs_ = self.covs[None,None,:,:,:].repeat(N,T-1,1,1,1)
        distribs = torch.distributions.MultivariateNormal(means_, covs_)
        log_local_evidence_T = distribs.log_prob(obs[:,1:,None,:].repeat(1,1,self.num_states,1))
        log_local_evidence = torch.cat([log_local_evidence_1, log_local_evidence_T], dim=1)
        return log_local_evidence

    def _maximization(self, gamma, paired_marginals, obs, obs_one):
        N, T, O = obs.shape
        # HMM params
        self.pi = gamma[:,0,:].sum(axis=0)/N
        N_jk = torch.sum(paired_marginals.reshape(-1,self.num_states, self.num_states), axis=0)+1e-6
        self.Q = N_jk/torch.sum(N_jk, axis=1, keepdims=True)

        # Emission densities
        # Init
        ## Means
        exp_means = torch.sum(gamma[:,0,:,None]*obs[:,0,None,:], axis=0)
        N_j = torch.sum(gamma[:,0,:], axis=0)+1e-6
        self.init_mean = exp_means/N_j[:,None]
        ## Covariances
        x_mu = obs[:,0,None,:] - self.init_mean[None,:,:]
        x_muXx_muT = torch.matmul(x_mu[:,:,:,None], x_mu[:,:,None,:])
        exp_cov = torch.sum(gamma[:,0,:,None, None]*x_muXx_muT, axis=0)
        self.init_cov = exp_cov/N_j[:,None,None] + torch.eye(self.obs_dim)[None,:,:].to(self.device)*1e-5
        # T >= 1
        ## Means
        _, _, c = obs_one.shape
        xtxt_1 = torch.matmul(obs[:,1:,:,None], obs_one[:,:,None,:])
        x_1xt_1 = torch.matmul(obs_one[:,:,:,None], obs_one[:,:,None,:])
        LHS = torch.sum(gamma[:,1:].reshape(N*(T-1),self.num_states,1,1)*xtxt_1.reshape(N*(T-1),1,O,c), axis=0)
        RHS = torch.sum(gamma[:,1:].reshape(N*(T-1),self.num_states,1,1)*x_1xt_1.reshape(N*(T-1),1,c,c), axis=0)
        self.means = torch.cat([torch.matmul(LHS[None,i], torch.linalg.inv(RHS[None,i])) for i in range(self.num_states)], dim=0)
        ## Covariances
        x_mu = obs[:,1:,:].reshape(N*(T-1),1, O) - torch.matmul(self.means[None,:,:,:], obs_one.reshape(N*(T-1), 1, -1, 1)).reshape(N*(T-1),self.num_states,O)
        x_muXx_muT = torch.matmul(x_mu[:,:,:,None], x_mu[:,:,None,:])
        exp_cov = torch.sum(gamma[:,1:].reshape(N*(T-1),self.num_states,1,1)*x_muXx_muT, axis=0)
        N_j = torch.sum(gamma[:,1:].reshape(-1,self.num_states), axis=0)+1e-6
        self.covs = exp_cov/N_j[:,None,None] + torch.eye(self.obs_dim)[None,:,:].to(self.device)*1e-5

    @torch.no_grad()
    def fit(self, obs, num_epochs=100, verbose=True, metric_mode='all', batch_size=10):
        N, T, D = obs.shape 
        poly = PolynomialFeatures(degree=self.coefs)
        obs_one = torch.from_numpy(poly.fit_transform(obs[:,:-1,:].reshape(N*(T-1),D)).reshape(N,T-1,-1))
        dataloader = DataLoader(TensorDataset(obs, obs_one), batch_size=batch_size, shuffle=True)
        pbar = tqdm(range(num_epochs))
        for i in pbar:
            for data in dataloader:
                obs_var = data[0].float().to(self.device)
                obs_one_var = data[1].float().to(self.device)
                local_evidence = self._compute_local_evidence(obs_var, obs_one_var)
                alpha, log_Z = self._forward(local_evidence)
                beta = self._backward(local_evidence, log_Z)
                gamma = self._compute_marginals(alpha, beta)
                paired_marginals = self._compute_paired_marginals(alpha, beta, local_evidence, log_Z)
                self._maximization(gamma, paired_marginals, obs_var, obs_one_var)
                loglikeli = self.LogLikelihood(gamma,paired_marginals,local_evidence)
                pbar.set_description("Log Likelihood: " + str(loglikeli))
        local_evidence = self._compute_local_evidence(obs_var, obs_one_var)
        alpha, log_Z = self._forward(local_evidence)
        beta = self._backward(local_evidence, log_Z)
        gamma = self._compute_marginals(alpha, beta)
        paired_marginals = self._compute_paired_marginals(alpha, beta, local_evidence, log_Z)
        return self.LogLikelihood(gamma,paired_marginals,local_evidence)

@torch.no_grad()
def sieve(num_states, obs_dim, obs, num_models=10, num_its=200, coefs=None, device='cpu', batch_size=100):
    log_likeli_list = []
    print("First Fit:")
    model_list = []
    for _ in tqdm(range(num_models)):
        model = PolyMSM(num_states, obs_dim, coefs=coefs, device=device)
        log_likeli_list.append((len(model_list), model.fit(obs, 10, verbose=False, batch_size=batch_size)))
        model_list.append(model)
    sorted_log_likeli = sorted(
        log_likeli_list,
        key=lambda t: t[1],
        reverse=True
    )
    log_likeli_list = []
    for (idx, _) in sorted_log_likeli[:5]:
        try:
            log_likeli_list.append((idx, model_list[idx].fit(obs, num_its, verbose=False, batch_size=batch_size)))
        except:
            continue
    best_model_id = sorted(
        log_likeli_list,
        key=lambda t: t[1],
        reverse=True
    )[0]
    return model_list[best_model_id[0]], best_model_id[1]

def function_dist(input_func, target_func, grid, degree):
    poly = PolynomialFeatures(degree=degree)
    obs_one = torch.from_numpy(poly.fit_transform(grid)).float()
    inferred_response = torch.matmul(input_func[None,:,:],obs_one[:,:,None])[:,:,0]
    target_response = torch.matmul(target_func[None,:,:],obs_one[:,:,None])[:,:,0]

    return torch.nn.functional.mse_loss(inferred_response, target_response)

def create_grid(dim, num_points, left_lim, right_lim, MC=True):
    if MC:
        grid = np.array([[np.random.uniform(left_lim, right_lim) for _ in range(dim)] for _ in range(1000)])
    else:
        grid_edges = [np.linspace(left_lim, right_lim, num_points) for _ in range(dim)]
        grid = np.stack(np.meshgrid(*grid_edges))
        grid.reshape(dim, -1).T
    return grid

def calc_dist_params(inferred_params, gt_params, dim_obs, mode='all', method='params', degree=None):
    # mode can be all|naive
    if mode=='all':
        return _calc_dist_params_all(inferred_params, gt_params, dim_obs, method, degree)
    else:
        return _calc_dist_params_naive(inferred_params, gt_params, dim_obs, method, degree)
    
def _calc_dist_params_naive(inferred_params, gt_params, dim_obs, method, degree=None):
    # Assuming all networks are approximated well, we can iterate to find the minimum distance
    # for each mode (search is O(K^2))
    K, *_ = inferred_params.shape
    idxs = list(range(K))
    permutation = []
    current_net = 0
    gt_params = torch.from_numpy(gt_params).float()
    while(len(idxs) != 0):
        min_dist = torch.inf
        current_gt_mode = gt_params[current_net]
        idx_max = idxs[0]
        for i in idxs:
            act_dist = torch.linalg.norm(inferred_params[i] - current_gt_mode)
            if (min_dist > act_dist):
                min_dist = act_dist
                idx_max = i
        current_net += 1
        permutation.append(idx_max)
        idxs.remove(idx_max)
    p_inferred_params = inferred_params[permutation]
    if method=='grid':
        grid = torch.from_numpy(create_grid(dim_obs, 10000, -1, 1)).float()
        dist = torch.tensor([function_dist(inferred_params[p], gt_params[idx], grid, degree) for idx,p in enumerate(permutation)]).sum()/K
    else:
        dist = torch.linalg.norm(p_inferred_params - gt_params)
    coef_dist = torch.linalg.norm(p_inferred_params - gt_params, axis=(0,1))
    return dist, coef_dist, permutation

def _calc_dist_params_all(inferred_params, gt_params, dim_obs, method, degree=None):
    # Dist params search over all permutations of K (search is O(K!))
    K, *_ = inferred_params.shape
    distances = []
    per_coef_distances = []
    permutations = np.array(list(itertools.permutations(range(K))))
    for permutation in permutations:
        p_inferred_params = inferred_params[permutation]
        if method=='grid':
            grid = torch.from_numpy(create_grid(dim_obs, 10000, -1, 1)).float()
            dist = torch.tensor([function_dist(inferred_params[p], gt_params[idx], grid, degree) for idx,p in enumerate(permutation)]).sum()/K
            distances.append(dist)
        else:
            distances.append(torch.linalg.norm(p_inferred_params - gt_params))
        per_coef_distances.append(torch.linalg.norm(p_inferred_params - gt_params, axis=(0,1)))
    distances = torch.tensor(distances)
    idx = torch.argmin(distances)
    dist = distances[idx]
    coef_dist = per_coef_distances[idx]
    perm = permutations[idx]
    return dist, coef_dist, perm

if __name__=="__main__":

    obs = np.load('observations_poly_N_1_T_100.npy')
    params = np.load('params_poly_N_1000_T_100_dim_5_state_10.npy')
    print(obs.shape)
    num_states = 3
    model = PolyMSM(num_states, 5, ar=True, coefs=2, device='cuda')
    model.fit(torch.from_numpy(obs).float(), num_its=200, metric_mode='naive')
    #print(params)
    #print(model.means)
    print(model.Q)
    #np.save("result_poly_N_1000_T_100_dim_5_state_10.npy", model.means.cpu())

    print(_calc_dist_params_naive(model.means.cpu(), params))
    print(_calc_dist_params_all(model.means.cpu(), params))

    

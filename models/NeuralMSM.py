import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from models.modules import MLP, CausalMLP
from models.MSM import MSM
from utils.benchmarks import benchmark_function_naive

class NeuralMSM(nn.Module, MSM):

    def __init__(self, num_states, obs_dim, hid_dim=8, device='cpu', lr=5e-3, causal=False, l1_penalty=0, l2_penalty=0, activation='cos', lag=1, gradient_clipping=None):
        
        # Parent initialisation
        nn.Module.__init__(self)
        MSM.__init__(self, num_states, obs_dim, device)

        self.causal = causal
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.lr = lr
        self.lag = lag
        self.gradient_clipping = gradient_clipping
        self.activation = activation
        
        if not self.causal:
            self.transitions = nn.ModuleList([MLP(obs_dim*self.lag, obs_dim, hid_dim, activation) for _ in range(self.num_states)]).to(device)
        else:
            self.transitions = nn.ModuleList([CausalMLP(obs_dim, hid_dim, activation, num_lags=self.lag) for _ in range(self.num_states)]).to(device)

        start_cov = 0.5
        self.covs = torch.nn.Parameter((torch.eye(self.obs_dim)[None,:,:]*start_cov).repeat(self.num_states,1,1).to(device))
        self.init_mean = torch.nn.Parameter((torch.rand(self.num_states, self.obs_dim*self.lag).to(device)-0.5)*2*10)
        self.init_cov = torch.nn.Parameter(((torch.rand(self.num_states,1,1)*torch.eye(self.obs_dim*self.lag)[None,:,:])*5).to(device))


        ## Scheduler and optimiser for NeuralMSM
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.5)


    def _compute_local_evidence(self ,obs):
        N, T, _ = obs.shape
        init_cov = torch.matmul(self.init_cov,self.init_cov.transpose(1,2)) + 1e-5*torch.eye(self.obs_dim*self.lag)[None,:,:].to(self.device)
        init_distrib_ = torch.distributions.MultivariateNormal(self.init_mean, init_cov)
        log_local_evidence_1 = init_distrib_.log_prob(obs[:,0:self.lag,:].reshape(N,1,1,self.obs_dim*self.lag).repeat(1,1,self.num_states,1))
        data_in = torch.cat([obs[:,i:-self.lag+i,None,:] for i in range(self.lag)],dim=2).reshape(N,T-self.lag,1,-1)
        means_ = torch.cat([self.transitions[i](data_in) for i in range(self.num_states)], dim=2)
        covs = torch.matmul(self.covs,self.covs.transpose(1,2)) + 1e-5*torch.eye(self.obs_dim)[None,:,:].to(self.device)
        distribs = [torch.distributions.MultivariateNormal(means_[:,:,i,:], covs[i,:,:]) for i in range(self.num_states)]
        log_local_evidence_T = torch.cat([distribs[i].log_prob(obs[:,self.lag:,:])[:,:,None] for i in range(self.num_states)], dim=2)
        return torch.cat([log_local_evidence_1, log_local_evidence_T], dim=1)

    def _maximization(self, gamma, paired_marginals, local_evidence):
        N, *_ = gamma.shape
        self.optimizer.zero_grad()
        # p(z_t|z_t-1)
        loss = -(gamma[:,:]*local_evidence[:,:]).sum()/N
        # Uncomment for gradient-based updates on pi and Q 
        # (for this remove .detach() from gamma and paired_marginals in posterior computations)
        # pi
        #loss -= (gamma[:,0,:]*torch.log(self.pi.softmax(-1)[None,:])).sum()/N
        # Q
        #Q = (self.Q[None,None,:,:].expand(N,T,-1,-1)).softmax(-1)
        #loss -= (paired_marginals*torch.log(Q[:,1:,:,:])).sum()/N
        if self.causal:
            for k in range(self.num_states):
                loss += self.l1_penalty*self.transitions[k].fc1_l1_reg() + self.l2_penalty*self.transitions[k].l2_reg()
        loss.backward()
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clipping)
        self.optimizer.step()
        # We use exact updates on Q and pi
        # Does not work well on small batch sizes
        with torch.no_grad():
            # HMM params
            self.gamma = gamma
            #self.pi = gamma[:,0,:].sum(dim=0)/N
            N_jk = torch.sum(paired_marginals.reshape(-1,self.num_states, self.num_states), dim=0)+1e-6
            self.Q = N_jk/torch.sum(N_jk, dim=1, keepdims=True)

    def fit(self, obs, num_epochs=100, verbose_path=None, batch_size=32, early_stopping=None, max_scheduling_steps=1):
        import sys
        if verbose_path is not None:
            params = np.load(verbose_path, allow_pickle=True).item()['arr']
        _, _, D = obs.shape
        dataloader = DataLoader(TensorDataset(obs), batch_size=batch_size, shuffle=True)
        # Training Loop implements a stopping-on-plateau schedule 
        # by reducing learning rate for some max_scheduling_steps
        count = 0
        loglikeli_list = []
        acc_list = []
        best_log_likeli = -np.inf
        scheduling_steps = 0
        model_plateaued = 0
        activate_trans = False
        pbar = tqdm(range(num_epochs))
        for i in pbar:
            if i >= 5 and not activate_trans:
                activate_trans = True
                model_plateaued = 0
            if model_plateaued==early_stopping and i >= 5:
                if scheduling_steps >= max_scheduling_steps:
                    break
                self.scheduler.step()
                model_plateaued = 0
                scheduling_steps += 1
            model_plateaued += 1
            for data in dataloader:
                if count%100 == 0:
                    if verbose_path is not None:
                        dist, perm = benchmark_function_naive(self.transitions.cpu().float(), params, self.lag*D)
                        self.transitions.to(self.device)
                        acc_list.append(dist)
                        print(dist, perm)
                        print(self.Q)
                obs_var = Variable(data[0].float(), requires_grad=True).to(self.device)
                local_evidence = self._compute_local_evidence(obs_var)
                with torch.no_grad():
                    alpha, log_Z = self._forward(local_evidence)
                    beta = self._backward(local_evidence, log_Z)
                    gamma = self._compute_marginals(alpha, beta)
                    paired_marginals = self._compute_paired_marginals(alpha, beta, local_evidence, log_Z)
                    loglikeli = self.LogLikelihood(gamma,paired_marginals,local_evidence)
                self._maximization(gamma, paired_marginals, local_evidence)
                if count%100 == 0:
                    loglikeli_list.append(loglikeli.cpu())
                pbar.set_description("Log Likelihood: {:0.4f}. Best: {:0.4f}".format(loglikeli, best_log_likeli))
                if best_log_likeli < loglikeli:
                    model_plateaued = 0
                    best_log_likeli = loglikeli
                count += 1
        with torch.no_grad():        
            local_evidence = self._compute_local_evidence(obs_var)
            alpha, log_Z = self._forward(local_evidence)
            beta = self._backward(local_evidence, log_Z)
            gamma = self._compute_marginals(alpha, beta)
            paired_marginals = self._compute_paired_marginals(alpha, beta, local_evidence, log_Z)
            loglikeli = self.LogLikelihood(gamma,paired_marginals,local_evidence)
        return loglikeli, loglikeli_list, acc_list
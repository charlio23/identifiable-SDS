from abc import ABC, abstractmethod
import torch

class MSM(ABC):
    def __init__(self, num_states, obs_dim, device='cpu'):
        self.num_states = num_states
        self.obs_dim = obs_dim
        self.device = device

        # Initialise MSM parameters
        self.pi = torch.ones(num_states).to(device)
        self.pi /= torch.sum(self.pi)
        self.Q = torch.ones(self.num_states, self.num_states).to(device)
        self.Q /= torch.sum(self.Q, axis=1, keepdims=True)

    @abstractmethod
    def _compute_local_evidence(self):
        raise NotImplementedError

    def _forward(self, local_evidence):
        N, T, _ = local_evidence.shape
        log_Z = torch.zeros((N,T)).to(self.device)
        log_alpha = torch.zeros((N, T, self.num_states)).to(self.device)
        log_prob = local_evidence[:,0,:] + torch.log(self.pi)
        log_Z[:,0] = torch.logsumexp(log_prob, dim=-1)
        log_alpha[:,0,:] = log_prob - log_Z[:,0,None]
        Q = (self.Q[None,None,:,:].expand(N,T,-1,-1)).transpose(2,3).log()
        for t in range(1, T):
            #log_prob = local_evidence[:,t,:] + torch.log(torch.matmul((Q.transpose(2,3))[:,t,:,:],alpha[:,t-1,:,None]))[:,:,0]
            log_prob = torch.logsumexp(local_evidence[:,t,:, None] + Q[:,t,:,:] + log_alpha[:,t-1,None,:], dim=-1) 
            
            log_Z[:,t] = torch.logsumexp(log_prob, dim=-1)
            log_alpha[:,t,:] = log_prob - log_Z[:,t,None]
        return log_alpha, log_Z

    def _backward(self, local_evidence, log_Z):
        N, T, _ = local_evidence.shape
        log_beta = torch.zeros((N, T, self.num_states)).to(self.device)
        Q = (self.Q[None,None,:,:].expand(N,T,-1,-1)).log()
        for t in reversed(range(1, T)):
            #beta_ = torch.matmul(Q[:,t,:,:], (torch.exp(local_evidence[:,t,:])*beta[:,t,:])[:,:,None])[:,:,0]
            beta_ = torch.logsumexp(Q[:,t,:,:] + local_evidence[:,t,None,:] + log_beta[:,t,None,:], axis=-1)
            log_beta[:,t-1,:] = beta_ - log_Z[:,t,None]
        return log_beta

    def _compute_marginals(self, log_alpha, log_beta):
        return (log_alpha + log_beta).exp().detach()

    def _compute_paired_marginals(self, log_alpha, log_beta, log_evidence, log_Z):
        B, T, _ = log_evidence.shape
        #alpha_beta_evidence = torch.matmul(alpha[:,:T-1,:,None], (beta*torch.exp(log_evidence))[:,1:,None,:])
        log_alpha_beta_evidence = log_alpha[:,:T-1,:,None] + log_beta[:,1:,None,:] + log_evidence[:,1:,None,:]
        Q = (self.Q[None,None,:,:].expand(B,T,-1,-1)).log()
        #paired_marginals = Q[:,1:,:,:]*(alpha_beta_evidence/torch.exp(log_Z[:,1:,None,None])).float()
        log_paired_marginals = Q[:,1:,:,:] + log_alpha_beta_evidence - log_Z[:,1:,None,None]
        return log_paired_marginals.exp().detach()

    def LogLikelihood(self, obs):
        N, T, D = obs.shape
        obs = obs.to(self.device)
        with torch.no_grad():
            local_evidence = self._compute_local_evidence(obs.to(self.device))
            _, log_Z = self._forward(local_evidence)

        return log_Z.sum()/(N*T*D)

    @abstractmethod
    def _maximization(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    

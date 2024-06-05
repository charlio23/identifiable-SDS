import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def get_trans_mat(size=3):
    """
    Returns transition matrix as a size x size matrix with 0.9 diagonals and 0.1 on another component.
    Example: get_trans_mat(3) returns
    Q = np.array([
        [0.9, 0.1, 0.0],
        [0.0, 0.9, 0.1],
        [0.1, 0.0, 0.9]
    ])
    """    
    Q = np.zeros((size,size))
    for i in range(Q.shape[0]-1):
        Q[i][i] = 0.9
        Q[i][i+1] = 0.1
    Q[-1][-1] = 0.9
    Q[-1][0] = 0.1
    return Q


def func_polynomial(x, features, degree=3):
    N, _ = x.shape

    poly = PolynomialFeatures(degree=degree)
    obs_one = poly.fit_transform(x[:,:])
    means_ = np.matmul(features[None,:,:], obs_one[:,:,None]).reshape(N,-1)
    return means_


def func_cosine(x, features):
    alphas, omegas, betas = features
    result = np.dot(omegas,x[:,None])[:,0]
    result = np.cos(result + betas)
    result = np.dot(alphas,result[:,None])[:,0]
    return result


def func_softplus(x, features):
    alphas, omegas, betas = features
    result = np.dot(omegas,x[:,None])[:,0]
    result = np.log(1 + np.exp(result + betas))
    result = np.dot(alphas,result[:,None])[:,0]
    return result

def func_leaky_relu(x, features):
    alphas, omegas, betas = features
    result = np.matmul(omegas[None,:,:],x[:,:,None])[:,:,0]
    result = result + betas[None,:]
    result = np.maximum(0.2*result, result)
    result = np.matmul(alphas[None,:,:],result[:,:,None])[:,:,0]
    return result

def func_cosine_with_sparsity(x, features):
    alphas, omegas, betas, adj_mat = features
    dim_obs = x.shape[-1]
    out = np.zeros_like(x)
    for i in range(dim_obs):
        input = x*adj_mat[i]
        result = np.matmul(omegas[:,i,:],input[:,None])[:,0]
        result = np.cos(result + betas[i])
        out[i] = np.dot(alphas[0,i,:],result)
    return out


def func_softplus_with_sparsity(x, features):
    alphas, omegas, betas, adj_mat = features
    dim_obs = x.shape[-1]
    out = np.zeros_like(x)
    for i in range(dim_obs):
        input = x*adj_mat[i]
        result = np.matmul(omegas[:,i,:],input[:,None])[:,0]
        result = np.log(1 + np.exp(result + betas[i]))
        out[i] = np.dot(alphas[0,i,:],result)
    return out


def sample_adj_mat(sparsity_prob, dim):
    edges = np.random.choice(
        [0,1], size=(dim, dim), p=[sparsity_prob, 1-sparsity_prob]
    )
    np.fill_diagonal(edges, 1)
    return edges
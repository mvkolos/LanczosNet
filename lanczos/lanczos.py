from numpy.linalg import qr, norm
from scipy.linalg import eigh_tridiagonal
import numpy as np
from tqdm import tqdm
import torch


def lanczos_algorithm(S, k, eps=0, re_ortho_rate=10):
    '''
    S - matrix
    k - number of columns < dim S
    re_ortho_rate - reorthogonalization every t steps
    
    returns: orthogonal basis in Krylov subspace
    '''
    n = S.shape[0]
    x = np.ones(n)

    b = 0
    q_prev = np.zeros(n)
    q = x/norm(x)
    Q = np.zeros((n, k))

    for j in tqdm(range(1, k+1)):
        Q[:, j-1] = q
        z = S@q
        v = q.T@z
        z = z - v*q - b*q_prev
        b = norm(z)

        if b < eps:
            break

        q_prev = q
        q = z/b
        if re_ortho_rate: 
            if j % re_ortho_rate == 0:
                Q = qr(Q)[0]

    T = Q.T@S@Q
    return Q, T


def lanczos_algorithm_rational(S, k, eps=0, sigma = 1.):
    '''
    S - matrix
    k - number of columns < dim S
    sigma - shift for inverse of S
    
    returns: orthogonal basis in rational Krylov subspace
 
    '''
    n = S.shape[0]
    x = np.ones(n)

    b = 0
    a = 0
    q_prev = np.zeros(n)
    w_prev = np.zeros(n)
    
    q = x/norm(x)
    w = x/norm(x)
    
    S_inv = np.linalg.inv(S+sigma*np.identity(n))
    w = S_inv@w
    
    Q = np.zeros((n, k))
    
    for j in tqdm(range(k//2)):
        Q[:, 2*j] = q 
        Q[:, 2*j+1] = w
        
        z = S@q
        t = S_inv@w
  
        v = q.T@z
        u = w.T@t
        
        z = z - v*q - b*q_prev
        t = t - u*w - a*w_prev
        
        b = norm(z)
        a = norm(t)

        if b < eps:
            break

        q_prev = q
        w_prev=w
        q = z/b
        w = t/a
        
        Q = qr(Q)[0]

    T = Q.T@S@Q
    return Q, T


def process_adjacency_matrix(A, k, make_S=True, ritz=True):
    '''
    A - np.array, adjacency_matrix
    k - number of lanczos_algorithm steps
    make_S - construct S from A
    ritz - use ritz decomposition (returns tridioganal T instead)
    '''
    if make_S:
        if type(A) == torch.Tensor:
            D = torch.sqrt(A.sum(0)).reshape(-1, 1)
            S = A / D / D.t()
        else:
            D = np.diag(A.sum(0))
            D = np.sqrt(np.linalg.inv(D))
            S = np.array(D@A@D)
    else:
        S = A.copy()

    if type(A) == torch.Tensor:
        Q, T = lanczos_algorithm(S.cpu().detach().numpy(), k)
    else:
        Q, T = lanczos_algorithm(S, k)

    if not ritz:
        return S, Q, T

    d = np.diagonal(T, 0)
    e = np.diagonal(T, -1)
    w, B = eigh_tridiagonal(d, e)

    return S, Q@B, np.diag(w)

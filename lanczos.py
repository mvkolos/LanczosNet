from numpy.linalg import qr, norm
import numpy as np
from tqdm import tqdm

def lanczos_algorithm(S, k, eps=0, re_ortho_rate = 10):
    '''
    S - matrix
    k - number of columns < dim S
    re_ortho_rate - reorthogonalization every t steps
    '''
    n = S.shape[0]
    x = np.ones(n)
    
    b = 0
    q_prev = np.zeros(n)
    q = x/norm(x)
    Q = np.zeros((n,k))
        
    for j in tqdm(range(1, k+1)):
        Q[:,j-1] = q
        z = S@q
        v = q.T@z
        z = z - v*q- b*q_prev
        b = norm(z)
        
        if b<eps:
            break
            
        q_prev = q
        q = z/b
        
        if j%re_ortho_rate==0:
            Q = qr(Q)[0]
    
    T = Q.T@S@Q
    return Q, T
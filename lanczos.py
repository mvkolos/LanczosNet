from numpy.linalg import qr, norm
from scipy.linalg import eigh_tridiagonal
import numpy as np
from tqdm import tqdm


def lanczos_algorithm(S, k, eps=0, re_ortho_rate=10):
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

        if j % re_ortho_rate == 0:
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
        D = np.diag(A.sum(0))
        D = np.sqrt(np.linalg.inv(D))
        S = np.array(D@A@D)
    else:
        S = A.copy()

    Q, T = lanczos_algorithm(S, k)

    if not ritz:
        return S, Q, T

    d = np.diagonal(T, 0)
    e = np.diagonal(T, -1)
    w, B = eigh_tridiagonal(d, e)

    return S, Q@B, np.diag(w)

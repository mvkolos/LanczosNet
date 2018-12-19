import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter


class SpectralConv(nn.Module):
    def __init__(self, in_features, out_features, k, short_scales, long_scales,
                 mlp_layers_number=3):
        super(SpectralConv, self).__init__()
        '''
        in_features - number of in features
        out_features - number of out features
        k - number of Ritz values, k<=n
        short_scales - scales for direct filters S^qX
        long_scales - scales of spectral filters VRV^T
        mlp_layers_number - number of layers in R values transformation
        '''
        self.short_scales = short_scales
        self.long_scales = long_scales
        self.mlp = []

        for _ in range(mlp_layers_number):
            self.mlp.append(nn.modules.Linear(k, k))
            self.mlp.append(nn.modules.ReLU())
        if len(self.mlp) > 0:
            self.mlp = nn.Sequential(*self.mlp[:-1])
        else:
            self.mlp = IdentityModule()

        self.W = Parameter(torch.Tensor(in_features*(len(short_scales) +
                                                     len(long_scales)), out_features))
        stdv = 1. / (in_features*(len(short_scales) + len(long_scales)))**0.5
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, X, S, V, R):
        '''
        X - featurs
        S - affinity matrix
        V - Q@B - Q from Lanczos, B from eigendecomposition
        R - Ritz values
        '''
        Y = X
        Z = Y
        features = []
        for l in range(1, self.short_scales[-1] + 1):
            Z = torch.mm(S, Z)
            if l in self.short_scales:
                features.append(Z)

        for i in self.long_scales:
            R_h = self.mlp(R**i)
            Z = torch.mm(V, torch.diag(R_h))
            Z = torch.mm(Z, V.t())
            Z = torch.mm(Z, Y)
            features.append(Z)

        return torch.mm(torch.cat(features, 1), self.W)


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs


class ExpKernel(nn.Module):
    def __init__(self, X, A, layers=None, e=10, learn_embedding=False):
        super().__init__()
        self.mlp = []
        if learn_embedding:
            self.X = Parameter(X)
        else:
            self.X = X

        self.A = Parameter(torch.Tensor(np.sign(A)))
        shape = X.shape[-1]

        if layers is not None:
            for layer in layers:
                self.mlp.append(nn.modules.Linear(shape, layer))
                self.mlp.append(nn.modules.ReLU())
                shape = layer
        if len(self.mlp) > 0:
            self.mlp = nn.Sequential(*self.mlp[:-1])
        else:
            self.mlp = IdentityModule()

        self.e = e

    def forward(self, input):
        Y = self.mlp(self.X)
        n = Y.size(0)
        norms = torch.sum(Y**2, dim=1, keepdim=True)
        norms_squares = (norms.expand(n, n) + norms.t().expand(n, n))
        distances_squared = torch.sqrt(1e-6 + norms_squares - 2 * Y.mm(Y.t()))
        A = torch.exp(-distances_squared/self.e)
        return torch.clamp(A*self.A, 0, 10)

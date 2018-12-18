import torch
from torch import nn


class SpectralConv(nn.Module):
    def __init__(self, k, short_scales, long_scales):
        super(SpectralConv, self).__init__()
        '''
        k - number of Ritz values, k<=n
        short_scales - scales for direct filters S^qX
        long_scales - scales of spectral filters VRV^T
        '''
        self.short_scales = short_scales
        self.long_scales = long_scales
        self.mlp = nn.Linear(k, k)

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

        return torch.cat(features, 1)


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs


class ExpKernel(nn.Module):
    def __init__(self, X, layers=None, e=10):
        super().__init__()
        self.input_features = X
        self.mlp = []
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
        Y = self.mlp(input)
        n = Y.size(0)
        norms = torch.sum(Y**2, dim=1, keepdim=True)
        norms_squares = (norms.expand(n, n) + norms.t().expand(n, n))
        distances_squared = torch.sqrt(1e-6 + norms_squares - 2 * Y.mm(Y.t()))
        A = torch.exp(-distances_squared/self.e)
        return A

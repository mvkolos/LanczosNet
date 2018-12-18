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
            Z = torch.mm(S,Z)
            if l in self.short_scales:
                features.append(Z)

        for i in self.long_scales:
            R_h = self.mlp(R**i)
            Z = torch.mm(V, torch.diag(R_h))
            Z = torch.mm(Z, V.t())
            Z = torch.mm(Z, Y)
            features.append(Z)
            
        return torch.cat(features, 1)
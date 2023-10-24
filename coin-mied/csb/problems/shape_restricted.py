import torch
import numpy as np
from csb.problems.problem_base import ProblemBase


class ShapeRestricted(ProblemBase):
    def __init__(self, *,
                 device,
                 data_X,
                 data_Y,
                 degree,
                 noise_sigma=0.1,
                 constraint='none',
                 scale_y=True):
        '''
        :param data_X: (M,), data to fit
        :param data_Y: (M,)
        '''
        assert(constraint in ['none', 'monoconcave'])
        super().__init__(device=device,
                         in_dim=degree + 2)
        self.stats = {}
        self.data_X = self.normalize(data_X.to(device),
                                     'x',
                                     record=True)
        self.data_Y = data_Y.to(device)
        self.scale_y = scale_y
        if scale_y:
            self.data_Y = self.normalize(self.data_Y,
                                         'y',
                                         record=True)
        self.degree = degree
        self.noise_sigma = noise_sigma
        self.constraint = constraint

        # Pre-compute binomial coefficients.

        # n!/k!(n-k!) = \prod_{i=1}^k (n-k+i)/i
        self.coef = []
        for k in range(degree + 1):
            numer = 1
            denom = 1
            for i in range(1, k + 1):
                denom *= i
                numer *= self.degree - k + i
            assert(numer % denom == 0)
            self.coef.append(numer // denom)
        self.coef = torch.tensor(self.coef,
                                 dtype=torch.float32,
                                 device=device) # (N+1,)


    def get_embed_dim(self):
        return self.degree + 2


    def normalize(self, X, coord, record=False):
        '''
        :param X: (M,)
        :param coord: either 'x' or 'y'
        '''
        if record:
            self.stats[coord] = {
                'min': X.min(),
                'max': X.max(),
                'std': torch.std(X, unbiased=True)
            }
        stat = self.stats[coord]
        lb = stat['min'] - stat['std']
        rb = stat['max'] + stat['std']
        X = (X - lb) / (rb - lb)
        return X


    def inverse_normalize(self, X, coord):
        '''
        :param X: (...)
        '''
        stat = self.stats[coord]
        lb = stat['min'] - stat['std']
        rb = stat['max'] + stat['std']
        return (rb - lb) * X + lb


    def bernstein(self, x, N):
        '''
        :param N: degree of the Bernstein polynomial
        :param x: (M,), points in [0, 1] to evaluate at
        :return: (M, N+1) evaluated vector
        '''
        x = x.unsqueeze(-1) # (M, 1)
        arange = torch.arange(N + 1, device=self.device).float()
        tmp1 = torch.pow(x, arange) # (M, N+1)
        tmp1[:, 0] = 1.0
        tmp2 = torch.pow(1 - x, N - arange) # (M, N+1)
        tmp2[:, -1] = 1.0
        tmp = tmp1 * tmp2 * self.coef # (M, N+1)
        return tmp


    def eval_log_p(self, beta):
        '''
        :param beta: (B, N+2)
        '''
        log_p = -(
            self.data_Y.unsqueeze(0) -
            self.predict(beta, self.data_X)
        ).square() / (2 * self.noise_sigma) # (B, M)
        log_p = log_p - beta.square().sum(-1).unsqueeze(-1) / 100 # weak prior
        log_p = log_p.sum(-1) # (B,)
        return log_p


    def predict(self, beta, x, original=False):
        '''
        :param beta: (B, N+2)
        :param x: (M,)
        :param original: whether the input/output need to retain the
        original scale
        :return: (B, M)
        '''
        if original:
            x = self.normalize(x, 'x')
        bern = self.bernstein(x, self.degree) # (M, N+1)
        dot = (beta[:, :-1].unsqueeze(1) * bern).sum(-1) # (B, M)
        dot = dot + beta[:, -1].unsqueeze(-1)
        if original and self.scale_y:
            dot = self.inverse_normalize(dot, 'y')
        return dot


    def eval_ineq(self, beta):
        '''
        :param beta: (B, N+2), beta in the paper
        '''
        if self.constraint == 'none':
            return None

        if self.constraint == 'monoconcave':
            beta = beta[:, :-1]
            # 0 <= ... <= beta_i - beta_{i-1} <= beta_{i-1} - beta_{i-2}
            tmp = (beta[:, :-2] - 2 * beta[:, 1:-1] + beta[:, 2:]) # (B, N-1)
            tmp = torch.cat([tmp, (beta[:, -2] - beta[:, -1]).unsqueeze(-1)],
                            -1) # (B, N)
            return tmp

        assert(False)

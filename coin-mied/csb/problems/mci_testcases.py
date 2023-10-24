'''
Monte Carlo Integration on 2D manifold test cases.
'''
from abc import ABC, abstractmethod
import torch
import numpy as np
import math

class MCITestcase(ABC):
    def __init__(self, bbox, f_py, f_torch):
        '''
        :param bbox: (2, 2) tensor, so that u is in
            [bbox[0, 0], bbox[0, 1]] x [bbox[1, 0], bbox[1, 1]]
        :param f_py: the function to integrate that takes (3,) -> scalar
        :param f_torch: like f_py but for torch and takes batches
        '''
        self.bbox = bbox
        self.f_py = f_py
        self.f_torch = f_torch
        self.gold = None


    def sample_bbox(self, batch_size, device):
        t = torch.rand([batch_size, 2], device=device) # (B, 2)
        bbox = self.bbox.to(device)
        lower = bbox[:, 0].unsqueeze(0) # (1, 2)
        upper = bbox[:, 1].unsqueeze(0) # (1, 2)
        return (upper - lower) * t + lower


    def sample_gt(self, batch_size, device):
        return self.phi_torch(self.sample_bbox(batch_size, device))


    def log_p(self, X):
        '''
        :param X: (B, 3)
        :return: (B,)
        '''
        return (X-X).sum(-1) # make sure gradients exist


    @abstractmethod
    def eval_ineq(self, X):
        '''
        :param X: (B, 3)
        :return: (B,)
        '''
        pass


    @abstractmethod
    def phi_py(self, x, y):
        '''
        Measure-preserving map from 2D rectangle to the surface.
        :return: (3,)
        '''
        pass


    @abstractmethod
    def phi_torch(self, X):
        '''
        Like phi_py but for torch and batched.
        :param X: (B, 2)
        :return: (B, 3)
        '''
        pass


    def get_gold(self):
        '''
        Numerical integration to compute the ground truth integral.
        '''
        if self.gold is None:
            from scipy import integrate
            tmp = integrate.dblquad(
                lambda y, x: self.f_py(*self.phi_py(x, y)),
                self.bbox[0, 0], self.bbox[0, 1],
                self.bbox[1, 0], self.bbox[1, 1],
                epsabs=1e-14, epsrel=1e-14)
            area = ((self.bbox[1, 1] - self.bbox[1, 0]) *
                    (self.bbox[0, 1] - self.bbox[0, 0]))
            self.gold = tmp[0] / area
            print(f'Gold: {self.gold} with abserr {tmp[1]}')
        return self.gold


    def compute_rel_err(self, samples):
        mci_result = self.f_torch(samples).mean()
        error = (mci_result - self.get_gold()).abs().item() / abs(self.get_gold())
        return error


class CylinderMCI(MCITestcase):
    def __init__(self, f_py, f_torch):
        super().__init__(bbox=torch.tensor([[-1.0, 1.0], [0.0, 2 * np.pi]]),
                         f_py=f_py, f_torch=f_torch)


    def eval_ineq(self, X):
        return torch.stack([
            (X[:, 0]**2+X[:, 1]**2-1)**2,
            X[:, 2] - 1,
            -1 - X[:, 2],
        ], -1)


    def phi_py(self, x, y):
        return (math.cos(y), math.sin(y), x)


    def phi_torch(self, X):
        return torch.stack([
            torch.cos(X[:, 1]),
            torch.sin(X[:, 1]),
            X[:, 0],
        ], -1)


g_test_fns = {
    'one': {
        'f_py': lambda x, y, z: 1,
        'f_torch': lambda X: torch.ones([X.shape[0]],
                                        device=X.device, dtype=X.dtype)
    },

    'f1': {
        'f_py': lambda x, y, z: math.sqrt((1+z)*(1-z)) * math.cos(x/2+y/3+z/5),
        'f_torch': (
            lambda X: ((1+X[:,2])*(1-X[:,2])).sqrt() *
            torch.cos(X[:,0]/2+X[:,1]/3+X[:,2]/5))
    },
    'f4': {
        'f_py': lambda x, y, z: (math.exp(-math.sqrt(x*x+y*y+z*z))/(1+x*x)*
                                 math.cos(1+x*x)*math.sin(1-y*y)*
                                 math.exp(abs(z))),
        'f_torch': lambda X: (torch.exp(-X.square().sum(-1).sqrt())/(1+X[:,0]**2)*
                              torch.cos(1+X[:,0]**2)*torch.sin(1-X[:,1]**2)*
                              torch.exp(X[:,2].abs()))
    }
}


def create_testcase(surface_name, f_name):
    f_dict = g_test_fns[f_name]
    if surface_name == 'cylinder':
        testcase = CylinderMCI(f_dict['f_py'], f_dict['f_torch'])
    return testcase

import torch
from csb.problems.problem_base import ProblemBase

'''
Monte Carlo integration on surfaces embeded in 3D.
'''
class MCI(ProblemBase):
    def __init__(self, *,
                 testcase,
                 device):
        super().__init__(in_dim=3,
                         device=device)
        self.testcase = testcase


    def get_embed_dim(self):
        return 2


    def eval_log_p(self, X):
        return self.testcase.log_p(X)


    def eval_ineq(self, X):
        return self.testcase.eval_ineq(X)


    def reparam(self, Z):
        tmp = torch.sigmoid(Z) # (B, 2)
        bbox = self.testcase.bbox.to(self.device)
        lower = bbox[:, 0].unsqueeze(0) # (1, 2)
        upper = bbox[:, 1].unsqueeze(0) # (1, 2)
        tmp = (upper - lower) * tmp + lower

        return self.testcase.phi_torch(tmp)


    def get_reparam(self):
        return {
            'reparam_fn': lambda Z: self.reparam(Z),
            'prior_sample_fn': (lambda B, device:
                                torch.randn([B, 2], device=device))
        }


    def sample_gt(self, batch_size):
        return self.testcase.sample_gt(batch_size, self.device)

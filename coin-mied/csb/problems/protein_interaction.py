import torch
import math
import numpy as np
from csb.problems.problem_base import ProblemBase


class ProteinInteraction(ProblemBase):
    def __init__(self, *,
                 device,
                 data_path=None,
                 num_protein=270,
                 rank=3,
                 rigid_prior=False,
                 synthesize=False,
                 link_fn='sigmoid'):
        '''
        Parameters are:
            U: mxp, Lambda: pxp diagonal, c: scalar
        '''
        self.p = rank
        self.m = num_protein
        self.rigid_prior = rigid_prior
        self.synthesize = synthesize
        if link_fn == 'probit':
            self.link_fn = lambda x: 0.5 + torch.special.erf(x) / 2
        else:
            assert(link_fn == 'sigmoid')
            self.link_fn = torch.sigmoid
        super().__init__(device=device, in_dim=self.p * (self.m + 1) + 1)

        if synthesize:
            params = self.sample_prior_impl(1, mode='rigid_prior')
            U, Lambda, c = self.unpack_params(params)
            L = torch.diag_embed(Lambda) # Bxpxp
            tmp = torch.bmm(U, torch.bmm(L, U.transpose(-2, -1))) # 1xmxm
            tmp = tmp + c.unsqueeze(-1).unsqueeze(-1) # 1xmxm
            tmp = self.link_fn(tmp) # 1xmxm
            self.data = (torch.rand([self.m, self.m], device=device)
                         < tmp.squeeze(0)).long()
            print('gt log_p: ', self.eval_log_p(params))
            self.gt_params = params.squeeze(0)
        else:
            assert(data_path is not None)
            self.data = torch.from_numpy(np.load(data_path)).to(device).long()


    def sample_prior(self, B):
        return self.sample_prior_impl(B,
                                      ('rigid_prior' if self.rigid_prior
                                       else 'normal'))


    def sample_prior_impl(self, B, mode='normal'):
        if mode == 'normal':
            return super().sample_prior(B)
        if mode == 'rigid_prior':
            U = torch.randn([B, self.m, self.p], device=self.device)
            U, _ = torch.linalg.qr(U, mode='reduced')
            assert(U.shape[-2] == self.m and U.shape[-1] == self.p)

            Lambda = torch.randn(
                [B, self.p], device=self.device) * math.sqrt(self.m)

            c = torch.randn([B, 1], device=self.device) * 10
        else:
            assert(mode == 'synth')
            U = torch.zeros([B, self.m, self.p], device=self.device)
            for i in range(self.p):
                U[:, i, i] = 1
            Lambda = torch.ones([B, self.p], device=self.device)
            c = torch.zeros([B, 1], device=self.device)
        return torch.cat([U.reshape(B, -1), Lambda, c], -1)


    def get_embed_dim(self):
        return self.p * self.m - self.p * (self.p + 1) / 2


    def unpack_params(self, P):
        '''
        :param P: BxD
        :return: (U, Lambda, c), where
            U: Bxmxp,
            Lambda: Bxp,
            c: B
        '''
        cur = 0
        U = P[:, :self.m * self.p].reshape(-1, self.m, self.p)
        cur += self.m * self.p
        Lambda = P[:, cur:cur+self.p]
        cur += self.p
        c = P[:, cur]
        assert(cur == P.shape[-1] - 1)
        return (U, Lambda, c)


    def eval_log_p(self, P, cross_entropy_only=False):
        U, Lambda, c = self.unpack_params(P)
        L = torch.diag_embed(Lambda) # Bxpxp
        tmp = torch.bmm(U, torch.bmm(L, U.transpose(-2, -1))) # Bxmxm
        tmp = tmp + c.unsqueeze(-1).unsqueeze(-1) # Bxmxm

        prob_1 = self.link_fn(tmp)
        prob_0 = 1 - self.link_fn(tmp)

        tmp = torch.where(self.data.unsqueeze(0) == 1, prob_1, prob_0) # Bxmxm
        tmp = (tmp + 1e-10).log()

        # Zero out diagonal.
        mask = torch.eye(self.m, device=self.device,
                         dtype=torch.bool).unsqueeze(0) # 1xmxm
        mask = torch.logical_not(mask)
        tmp = torch.where(mask, tmp, torch.zeros_like(tmp)) # Bxmxm

        log_p = tmp.sum(-1).sum(-1) # B
        if cross_entropy_only:
            return -log_p
        log_p = log_p - Lambda.square().sum(-1) / (2 * self.m)
        log_p = log_p - c.square() / 200


        return log_p


    def eval_pred(self, P):
        U, Lambda, c = self.unpack_params(P)
        L = torch.diag_embed(Lambda) # Bxpxp
        tmp = torch.bmm(U, torch.bmm(L, U.transpose(-2, -1))) # Bxmxm
        tmp = tmp + c.unsqueeze(-1).unsqueeze(-1) # Bxmxm

        prob_1 = self.link_fn(tmp) # Bxmxm
        return prob_1


    def eval_eq(self, P):
        U, _, _ = self.unpack_params(P)
        tmp = torch.bmm(U.transpose(-2, -1), U) # Bxpxp
        I = torch.eye(self.p).to(U).unsqueeze(0) # 1xpxp
        tmp = (tmp - I).reshape(U.shape[0], -1) # Bx(p*p)
        return tmp


    def custom_eval(self, samples):
        return {
            'log_p': self.eval_log_p(samples).detach().cpu(),
            'ce': self.eval_log_p(samples, cross_entropy_only=True).detach().cpu(),
            'data': self.data.detach().cpu(),
            'pred': self.eval_pred(samples).detach().cpu(),
            'gt_params': self.gt_params.detach().cpu(),
        }

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from csb.problems.problem_base import ProblemBase


class MovieLensDataset(torch.utils.data.Dataset):
    def __init__(self, data_path,
                 user_lim=None,
                 movie_lim=None,
                 device=torch.device('cpu')):
        cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        df = pd.read_csv(data_path, sep='::', names=cols).drop(
            columns=['timestamp']
        ).astype(int)

        self.user_id = torch.from_numpy(df[['user_id']].values).squeeze(-1) - 1
        self.movie_id = torch.from_numpy(df[['movie_id']].values).squeeze(-1) - 1
        self.rating = torch.from_numpy(df[['rating']].values).squeeze(-1)
        # Normalize ratings to be [0, 1].
        self.rating = (self.rating.float() - 1) / 4

        self.num_user = self.user_id.max() + 1
        self.num_movie = self.movie_id.max() + 1

        if user_lim is not None and user_lim > 0:
            assert(movie_lim is not None and movie_lim > 0)
            # Create a subset of the dataset with first movie_lim movies and
            # select users with most number of ratings for these movies.
            rating_count = torch.zeros(self.num_user, dtype=int)
            tmp = self.user_id[self.movie_id < movie_lim]
            rating_count.scatter_add_(0, tmp,
                                      torch.ones_like(tmp).to(rating_count))

            _, user_inds = torch.topk(rating_count, user_lim)
            user_inds = torch.sort(user_inds)[0]
            user_mask = torch.zeros(self.num_user, dtype=torch.bool)
            user_mask[user_inds] = True

            should_keep = torch.logical_and(user_mask[self.user_id],
                                            self.movie_id < movie_lim)
            self.user_id = self.user_id[should_keep]
            self.user_id = torch.searchsorted(user_inds, self.user_id)
            self.movie_id = self.movie_id[should_keep]
            self.rating = self.rating[should_keep]
            print('Kept {} ratings after filtering.'.format(self.rating.shape[0]))
            self.num_user = user_lim
            self.num_movie = movie_lim


        self.user_id = self.user_id.to(device)
        self.movie_id = self.movie_id.to(device)
        self.rating = self.rating.to(device)
        self.num_rating = len(self)


    def __len__(self):
        return self.user_id.shape[0]


    def __getitem__(self, i):
        return (self.user_id[i], self.movie_id[i], self.rating[i])


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, num_user, num_movie, rank,
                 noise=0.2,
                 vis_ratio=0.5,
                 seed=1234,
                 device=torch.device('cpu')):
        rs = np.random.RandomState(seed=seed)
        U = torch.from_numpy(rs.randn(num_user, rank)) # Nxr
        U, _ = torch.linalg.qr(U, mode='reduced')
        V = torch.from_numpy(rs.randn(num_movie, rank)) # Mxr
        V, _ = torch.linalg.qr(V, mode='reduced')
        S = torch.from_numpy(rs.randn(rank)) # r
        # Issue with random generation: ratings will all be near 0.5


        tmp = torch.sigmoid(U @ torch.diag(S) @ V.T) # NxM
        tmp = tmp + rs.randn(num_user, num_movie) * (noise ** 2)

        self.Y = tmp

        num_vis = int((num_user * num_movie) * vis_ratio)
        inds = rs.choice(num_user * num_movie, num_vis, replace=False)
        self.user_id = []
        self.movie_id = []
        self.rating = []
        for k in inds:
            i = k // num_movie
            j = k % num_movie
            self.user_id.append(i)
            self.movie_id.append(j)
            self.rating.append(self.Y[i, j])

        self.user_id = torch.tensor(self.user_id).to(device)
        self.movie_id = torch.tensor(self.movie_id).to(device)
        self.rating = torch.tensor(self.rating).float().to(device)

        self.U = U
        self.V = V
        self.S = S
        self.num_user = num_user
        self.num_movie = num_movie
        self.num_rating = self.user_id.shape[0]


    def __len__(self):
        return self.user_id.shape[0]


    def __getitem__(self, i):
        return (self.user_id[i], self.movie_id[i], self.rating[i])


class CollaborativeFiltering(ProblemBase):
    def __init__(self, *,
                 device,
                 rank,
                 noise,
                 data_path=None,
                 movie_lim=100,
                 user_lim=100):
        if data_path is not None:
            self.dataset = MovieLensDataset(data_path=data_path,
                                            device=device,
                                            movie_lim=movie_lim,
                                            user_lim=user_lim)
        else:
            self.dataset = ToyDataset(num_user=user_lim,
                                      num_movie=movie_lim,
                                      rank=rank,
                                      noise=noise,
                                      device=device)
        self.r = rank
        self.noise = noise
        self.N = self.dataset.num_user
        self.M = self.dataset.num_movie
        self.dim = self.r * (self.N + self.M + 1)
        super().__init__(device=device,
                         in_dim=self.dim)


    def get_embed_dim(self):
        return self.dim - self.r * (self.r + 1)


    def unpack_params(self, P):
        '''
        :param P: BxD
        :return: (U, V, S), where
            U: BxNxr,
            V: BxMxr,
            S: Bxr
        '''
        cur = 0
        U = P[:, :self.N * self.r].reshape(-1, self.N, self.r)
        cur += self.N * self.r
        V = P[:, cur:cur+self.M * self.r].reshape(-1, self.M, self.r)
        cur += self.M * self.r
        S = P[:, cur:cur+self.r]
        cur += self.r
        assert(cur == P.shape[-1])
        return (U, V, S)


    def eval_log_p(self, P, nmae=False):
        U, V, S_flat = self.unpack_params(P)
        # S_flat: Bxr
        S = torch.diag_embed(S_flat) # Bxrxr

        # For now use the entire dataset.
        u_id = self.dataset.user_id # Z
        m_id = self.dataset.movie_id # Z
        rating = self.dataset.rating # Z
        U_ex = U[:, u_id, :] # BxZxr
        V_ex = V[:, m_id, :] # BxZxr

        tmp = (U_ex * S_flat.unsqueeze(1) * V_ex).sum(-1) # BxZ
        tmp = torch.sigmoid(tmp) # BxZ
        tmp = tmp - rating.unsqueeze(0)
        if nmae:
            return (4 * tmp).abs().mean(-1) / 1.6 # specifically for ML
        log_p = -tmp.square() / (self.noise ** 2) # BxZ
        log_p = log_p.sum(-1) # B

        # Account for the prior for S.
        log_p += -(S.square() / (2 * self.N * self.M)).sum(-1).sum(-1)

        return log_p


    def eval_eq(self, P):
        U, V, _ = self.unpack_params(P)
        I = torch.eye(self.r).to(U).unsqueeze(0) # 1xrxr

        tmp = torch.cat([
            (U.transpose(-2, -1) @ U - I).reshape(U.shape[0], -1),
            (V.transpose(-2, -1) @ V - I).reshape(V.shape[0], -1)
        ], -1)
        return tmp


    def eval_pred(self, P):
        # This can be expensive if N * M is large!
        U, V, S = self.unpack_params(P)
        S = torch.diag_embed(S) # Bxrxr
        pred = torch.sigmoid(U @ S @ V.transpose(-2, -1)) # BxNxM
        return pred


    def custom_eval(self, samples):
        U, V, S = self.unpack_params(samples)

        result = {
            'dataset': {
                'user_id': self.dataset.user_id,
                'movie_id': self.dataset.movie_id,
                'rating': self.dataset.rating,
            },
            'U': U,
            'V': V,
            'S': S,
            'log_p': self.eval_log_p(samples),
            'nmae': self.eval_log_p(samples, nmae=True),
        }
        if self.dataset.num_user * self.dataset.num_movie < 100000:
            result['pred'] = self.eval_pred(samples)

        return result



    def custom_post_step(self, samples):
        log_p = self.eval_log_p(samples)
        nmae = self.eval_log_p(samples, nmae=True)
        return {
            'log_p_avg': log_p.mean(),
            'log_p_max': log_p.max(),
            'NMAE_avg': nmae.mean(),
            'NMAE_best': nmae.min(),
        }


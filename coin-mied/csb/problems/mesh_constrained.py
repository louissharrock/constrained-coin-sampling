import torch
from csb.problems.problem_base import ProblemBase
from geomlib.tri_mesh import TriMesh

class MeshConstrained(ProblemBase):
    def __init__(self, *,
                 device,
                 tri_mesh,
                 log_p_fn):
        '''
        :param tri_mesh: an instance of TriMesh
        '''
        super().__init__(device=device,
                         in_dim=3)
        self.tri_mesh = tri_mesh
        self.log_p_fn = log_p_fn


    def get_embed_dim(self):
        return 2


    def eval_constraints(self, P):
        '''
        :param P: (B, 3)
        :return: (B,)
        '''
        assert(P.is_cuda)
        dist, face_idx = self.tri_mesh.compute_udf_gpu(P)
        return dist.square()


    def eval_log_p(self, P):
        return self.log_p_fn(P)

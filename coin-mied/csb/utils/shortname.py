from csb.solvers.mied import MIED, CoinMIED
from csb.solvers.svgd import SVGD, CoinSVGD
from csb.solvers.ksdd import KSDD, CoinKSDD
from csb.solvers.ipd import IPD, CoinIPD
from csb.solvers.dynamic_barrier import DynamicBarrier
from csb.solvers.no_op_projector import NoOpProjector

g_methods = [
    (MIED, 'MIED'),
    (SVGD, 'SVGD'),
    (KSDD, 'KSDD'),
    (IPD, 'IPD'),
    (CoinMIED, 'coin-mied'),
    (CoinSVGD, 'CoinSVGD'),
    (CoinKSDD, 'CoinKSDD'),
    (CoinIPD, 'CoinIPD'),
]

g_projectors = [
    (DynamicBarrier, 'DB'),
    (NoOpProjector, 'NOOP'),
]

def convert_method_cls_to_str(method_cls):
    for pr in g_methods:
        if pr[0] == method_cls:
            return pr[1]
    raise Exception(f'Unregisted method class {method_cls}!')


def convert_method_str_to_cls(method_str):
    for pr in g_methods:
        if pr[1] == method_str:
            return pr[0]
    raise Exception(f'Unregisted method str {method_str}!')


def convert_projector_cls_to_str(projector_cls):
    for pr in g_projectors:
        if pr[0] == projector_cls:
            return pr[1]
    raise Exception(f'Unregisted projector class {projector_cls}!')

def convert_projector_str_to_cls(projector_str):
    for pr in g_projectors:
        if pr[1] == projector_str:
            return pr[0]
    raise Exception(f'Unregisted projector str {projector_str}!')

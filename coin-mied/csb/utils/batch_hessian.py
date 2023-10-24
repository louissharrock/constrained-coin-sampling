import torch
from csb.utils.batch_jacobian import compute_jacobian

def compute_hessian(func, inputs):
    '''
    Compute Hessianmatrices in batch.

    :param func: (B, D) -> (B,)
    :param inputs: (B, D)
    :returns: (B, D, D)
    '''
    outputs = func(inputs) # (B,)
    # grad = torch.autograd.grad(outputs=outputs, inputs=inputs,
    #                            grad_outputs=torch.ones(inputs.shape[:-1], device=inputs.device),
    #                            create_graph=True, retain_graph=True)[0] # (B, D)
    grad = compute_jacobian(outputs.unsqueeze(-1), inputs).squeeze(-2) # (B, D)

    result = compute_jacobian(grad, inputs)

    return result


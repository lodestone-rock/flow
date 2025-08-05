import torch
import numpy as np


def cosine_optimal_transport(X, Y, backend="auto"):
    """
    Compute optimal transport between two sets of vectors using cosine distance.

    Parameters:
    X: torch.Tensor of shape (n, d)
    Y: torch.Tensor of shape (m, d)
    backend: str, optional (default='auto')
        'auto': Try CUDA implementation first, fall back to SciPy if unavailable
        'cuda': Force CUDA implementation (will raise error if unavailable)
        'scipy': Force SciPy implementation

    Returns:
    C: cost matrix of shape (n, m)
    matching_pairs: tuple of tensors (row_indices, col_indices)
    """
    # Normalize input vectors
    X_norm = X / torch.norm(X, dim=1, keepdim=True)
    Y_norm = Y / torch.norm(Y, dim=1, keepdim=True)

    # Compute cost matrix using matrix multiplication (cosine similarity)
    C = -torch.mm(X_norm, Y_norm.t())  # negative because we want to minimize distance

    if backend == "scipy":
        return _scipy_assignment(C)
    elif backend == "cuda":
        return _cuda_assignment(C)
    else:  # 'auto'
        try:
            return _cuda_assignment(C)
        except (ImportError, RuntimeError) as e:
            print(f"Falling back to SciPy implementation: {str(e)}")
            return _scipy_assignment(C)


def _cuda_assignment(C):
    """Use the CUDA implementation for assignment"""
    from torch_linear_assignment import batch_linear_assignment
    from torch_linear_assignment import assignment_to_indices

    original_device = C.device
    C_cpu = C.cpu().to(torch.float32).unsqueeze(dim=0)
    assignment = batch_linear_assignment(C_cpu)
    row_indices, col_indices = assignment_to_indices(assignment)
    row_indices = row_indices.to(original_device)
    col_indices = col_indices.to(original_device)
    matching_pairs = (row_indices, col_indices)

    return C, matching_pairs


def _scipy_assignment(C):
    """Use the SciPy implementation for assignment"""
    from scipy.optimize import linear_sum_assignment

    # Convert to numpy for scipy
    C_np = C.to(torch.float32).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(C_np)

    # Convert back to PyTorch tensors to match the CUDA implementation format
    matching_pairs = (
        torch.tensor([row_ind], device=C.device),
        torch.tensor([col_ind], device=C.device),
    )

    return C, matching_pairs

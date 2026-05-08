"""
Module: fusion.py

Description:
Main fusion routine to combine multiple mascon solutions into a single, 
optimally weighted solution.

Functions:
- build_design_matrix
- build_weights
- fusion_setup
- fusion_solve
- get_block_rms

Author: R McGirr 2026-04
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

def build_design_matrix(target, insols):
    """
    Build sparse design matrix mapping target mascons to product mascons.

    Each row represents one product mascon.
    Each column represents one target mascon.
    Entries are fractional area overlaps.

    Parameters
    ----------
    target : object with attributes
        mascon_id : ndarray (ngrid,)
        area      : ndarray (ngrid,)
        nmascons  : int

    insols : dict of objects, each with attributes
        mascon_id : ndarray (ngrid,)
        area      : ndarray (ngrid,)
        offset    : int   (row offset in A)
        nmascons  : int

    Returns
    -------
    A : scipy.sparse.csr_matrix (nobs, nparams)
    row_slices : dict of slices for each product
    """

    nparams = target.nmascons

    # --------------------------------------------------
    # Calculate offsets automatically
    # --------------------------------------------------
    offset = 0
    for insol in insols.values():
        insol.offset = offset
        offset += insol.nmascons

    nobs = offset

    rows = []
    cols = []
    vals = []

    row_slices = {}

    for name, insol in insols.items():
        start = insol.offset
        stop  = start + insol.nmascons

        row_slices[name] = slice(start, stop)

        # Loop over target mascons
        for target_id in range(nparams):

            target_mid = target_id + 1   # convert index → mascon_id

            grid_idx = (target.mascon_id == target_mid)

            # raise an error if we can't find any grid points
            if not np.any(grid_idx):
                raise ValueError(f"No grid cells found for target mascon_id {target_mid}")

            # product mascons overlapping this target mascon
            product_ids = insol.mascon_id[grid_idx]
            overlap_area = insol.area[grid_idx]

            for pid in np.unique(product_ids):

                overlap_sum = np.sum(overlap_area[product_ids == pid])
                frac = overlap_sum / insol.mascons['area'][pid - 1]

                rows.append(start + pid - 1)
                cols.append(target_id)
                vals.append(frac)

    A = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(nobs, nparams)
    )

    return A, row_slices

def build_weights(insols, nobs, row_slices, block_scalars, use_area):

    W_vec = np.zeros(nobs)
    for name, sl in row_slices.items():

        insol = insols[name]

        if use_area:
            area = insol.mascons['area']
            weights = area / np.sum(area)
        else:
            weights = np.ones(insol.nmascons) / insol.nmascons

        W_vec[sl] = block_scalars[name] * weights

    # optional normalisation so mean weight ~1
    W_vec /= np.mean(W_vec)

    return W_vec

def fusion_setup(A, W_vec):
    """
    Precompute reusable matrices for fusion and SE.
    """

    W = sparse.diags(W_vec, format='csc')

    AtW = A.T @ W
    N = AtW @ A

    solver = splu(N.tocsc())

    # Compute inverse diagonal once
    nparams = N.shape[0]
    N_inv_diag = np.zeros(nparams)

    for i in range(nparams):
        e = np.zeros(nparams)
        e[i] = 1.0
        col = solver.solve(e)
        N_inv_diag[i] = col[i]

    return solver, AtW, N_inv_diag

def fusion_solve(A, W_vec, solver, AtW, N_inv_diag, row_slices, b_vec):

    nobs = len(b_vec)
    nparams = len(N_inv_diag)

    rhs = AtW @ b_vec
    x_hat = solver.solve(rhs)

    res = b_vec - A @ x_hat

    sigma0_sq = np.sum(W_vec * res**2) / (nobs - nparams)

    SE = np.sqrt(sigma0_sq * N_inv_diag)

    return x_hat, SE, res, sigma0_sq

def get_block_rms(res, W_vec, row_slices):

    block_rms = {}

    for name, sl in row_slices.items():

        r_block = res[sl]
        w_block = W_vec[sl]

        block_rms[name] = np.sqrt(np.sum(w_block * r_block**2) / np.sum(w_block))

    return block_rms

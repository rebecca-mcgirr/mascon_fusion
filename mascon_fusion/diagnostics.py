"""
Module: diagnostics.py

Description:
Diagnostic routines for analyzing fusion results.

Functions:
- compute_region_timeseries
- eigen_diagnostics
- plot_eigen_diagnostics
- plot_fusion_diagnostics
- plot_solution_uncertainty
- plot_plot_residuals

Author: R McGirr 2026-04
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import sparse
from scipy.sparse.linalg import eigsh

def compute_region_timeseries(grid, mask, units='m'):
    """
    Compute area-weighted mean EWH time series for region.
    """

    area = grid.area
    ewh = grid.ewh  # (nt, nlat, nlon)
    if units == 'm':
        ts = np.nansum((ewh[:, mask] * area[mask]),axis=1) / np.nansum(area[mask])
    elif units == 'Gt':
        # get volume in m^3 then convert to Gt assuming density of 1000 kg/m^3
        ts = np.nansum((ewh[:, mask] * area[mask]),axis=1) / 1e9
    else:
        raise ValueError(f"Units {units} not recognized. Use 'm' or 'Gt'.")

    return ts

def eigen_diagnostics(A, W_vec, k_small=30, k_large=30):

    W = sparse.diags(W_vec)

    N = A.T @ W @ A
    #print('computed normal')
    # smallest eigenvalues
    vals_small = eigsh(N, k=k_small, tol=1e-2, which='SM',
                       return_eigenvectors=False)
    #print('computed small')
    # largest eigenvalues
    vals_large = eigsh(N, k=k_large, tol=1e-2, which='LM',
                       return_eigenvectors=False)
    #print('computed large')
    evals = np.sort(np.concatenate([vals_small, vals_large]))

    lam_min = evals[0]
    lam_max = evals[-1]
    cond = lam_max / lam_min

    return evals, lam_min, lam_max, cond

def plot_eigen_diagnostics(A, W_vec, row_slices, exp_id, k_small=30, k_large=30):
    evals, lam_min, lam_max, cond = eigen_diagnostics(A, W_vec, k_small=k_small, k_large=k_large)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot weight vector
    axes[0].plot(W_vec)
    axes[0].set_title(f'Weight Vector ({exp_id})')
    axes[0].set_xlabel('Observation index')
    axes[0].set_ylabel('Weight')
    # add vertical lines for block boundaries
    for name, sl in row_slices.items():
        axes[0].axvline(sl.start, color='r', linestyle='--', linewidth=0.8)
        axes[0].text(sl.start + (sl.stop - sl.start) / 2, axes[0].get_ylim()[1], name,
                    ha='center', va='top', fontsize=9, color='r')

    # Plot small eigenvalues
    axes[1].plot(evals[:k_small], 'o-', markersize=4)
    axes[1].set_title(f'Small Eigenvalues (first {k_small})')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_yscale('log')

    # Plot large eigenvalues
    axes[2].plot(evals[-k_large:], 'o-', markersize=4, color='orange')
    axes[2].set_title(f'Large Eigenvalues (last {k_large})')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Eigenvalue')

    fig.suptitle(f'Condition number: {cond:.2f}  |  λ_min={lam_min:.4f}, λ_max={lam_max:.4f}', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_fusion_diagnostics(target, sigma0, block_rms, SE_all, bins=30):
    x = target.decyear
    line_colors = {'JPL': 'tab:blue', 'GSFC': 'tab:orange', 'CSR': 'tab:green', 'ANU': 'tab:red'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), dpi=300, sharex=False, constrained_layout=True)

    # -------------------------
    # (a) sigma0
    # -------------------------
    axes[0].plot(x, sigma0, color='black', lw=1.8)
    axes[0].set_title('(a) Fusion variance factor ($\\sigma_0$)', fontsize=12, pad=8)
    axes[0].set_xlabel('Year', fontsize=11)
    axes[0].set_ylabel('$\\sigma_0$', fontsize=11)
    axes[0].grid(True, which='major', alpha=0.25)
    axes[0].minorticks_on()
    axes[0].grid(True, which='minor', alpha=0.12)

    # -------------------------
    # (b) Block RMS
    # -------------------------
    for key, vals in block_rms.items():
        axes[1].plot(
            x, vals,
            lw=1.8,
            label=key,
            color=line_colors.get(key, None)
        )

    axes[1].set_title('(b) Block RMS by product', fontsize=12, pad=8)
    axes[1].set_xlabel('Year', fontsize=11)
    axes[1].set_ylabel('RMS residual (m EWH)', fontsize=11)
    axes[1].grid(True, which='major', alpha=0.25)
    axes[1].minorticks_on()
    axes[1].grid(True, which='minor', alpha=0.12)
    axes[1].legend(frameon=False, ncol=2, fontsize=9, loc='upper left')

    # -------------------------
    # (c) Histogram of mean SE (per mascon)
    # -------------------------

    mean_se = np.nanmean(SE_all, axis=0)
    mse_mask = np.isfinite(mean_se)
    axes[2].hist(mean_se[mse_mask], bins=bins, color='tab:purple', alpha=0.85, edgecolor='white')
    axes[2].set_title('(c) Histogram of mean SE', fontsize=12, pad=8)
    axes[2].set_xlabel('Mean SE (m EWH)', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].grid(True, which='major', alpha=0.25)
    axes[2].minorticks_on()
    axes[2].grid(True, which='minor', alpha=0.12)
    # add dashed line for overall mean SE
    overall_mean_se = np.nanmean(mean_se)
    axes[2].axvline(overall_mean_se, color='black', linestyle='--', lw=1.5)
    # add as text annotation
    xshift = (axes[2].get_xlim()[1] - axes[2].get_xlim()[0]) * 0.04
    axes[2].text(
        overall_mean_se-xshift, axes[2].get_ylim()[1]*0.85,
        f'Mean SE={overall_mean_se:.4f} m',
        rotation=90, verticalalignment='center', 
        fontsize=9, color='black'
    )

    for ax in axes:
        ax.tick_params(axis='both', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.show()

def plot_solution_uncertainty(target, decyear):
    # Publication-style map: fused solution and standard error at one epoch

    # find index nearest to decyear
    t_plot = np.argmin(np.abs(target.decyear - decyear))
    year_label = target.decyear[t_plot]

    # count number of inputs contributing to each mascon at this epoch
    n_inputs = np.sum(np.isfinite([target.mascons['block_rms'][name][t_plot] 
                      for name in target.mascons['block_rms'].keys()]))

    ewh_map = target.ewh[t_plot]
    se_map = target.SE[t_plot]

    # Robust color limits
    ewh_lim = np.nanpercentile(np.abs(ewh_map), 95)
    ewh_lim = max(ewh_lim, 0.1)
    se_lim = np.nanpercentile(se_map, 95)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11
    })

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 4.8),
        dpi=180,
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True
    )

    # (a) Fused solution
    ax0 = axes[0]
    pcm0 = ax0.pcolormesh(
        target.lonv, target.latv, ewh_map,
        transform=ccrs.PlateCarree(),
        cmap="RdBu",
        vmin=-ewh_lim, vmax=ewh_lim,
        shading="auto"
    )
    ax0.coastlines(linewidth=0.6)
    ax0.set_global()
    ax0.set_title(f"(a) Fused solution (EWH), t={year_label:.2f}, n_inputs={n_inputs}")
    cb0 = fig.colorbar(pcm0, ax=ax0, orientation="horizontal", pad=0.05, fraction=0.05)
    cb0.set_label("Equivalent water height (m)")

    # (b) Standard error
    ax1 = axes[1]
    pcm1 = ax1.pcolormesh(
        target.lonv, target.latv, se_map,
        transform=ccrs.PlateCarree(),
        cmap="magma",
        vmin=0, vmax=se_lim,
        shading="auto"
    )
    ax1.coastlines(linewidth=0.6)
    ax1.set_global()
    ax1.set_title(f"(b) Standard error, t={year_label:.2f}, n_inputs={n_inputs}")
    cb1 = fig.colorbar(pcm1, ax=ax1, orientation="horizontal", pad=0.05, fraction=0.05)
    cb1.set_label("Standard error (m)")

    fig.suptitle("Mascon Fusion: Solution and Uncertainty", y=1.02, fontsize=13)
    plt.show()

def plot_block_residuals(target, decyear):

    # get index nearest to decyear
    t_plot = np.argmin(np.abs(target.decyear - decyear))

    year_label = target.decyear[t_plot]
    products = list(target.residuals.keys())

    # Robust color limits across all products
    all_res = np.concatenate([target.residuals[name][t_plot].ravel() for name in products])
    res_lim = np.nanpercentile(np.abs(all_res), 95)
    res_lim = max(res_lim, 0.01)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 8),
        dpi=180,
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True
    )

    for i, name in enumerate(products):
        ax = axes.flat[i]
        res_map = target.residuals[name][t_plot]

        pcm = ax.pcolormesh(
            target.lonv, target.latv, res_map,
            transform=ccrs.PlateCarree(),
            cmap="RdBu",
            vmin=-res_lim, vmax=res_lim,
            shading="auto"
        )
        ax.coastlines(linewidth=0.6)
        ax.set_global()
        rms_val = target.mascons['block_rms'][name][t_plot]
        rms_str = f", RMS={rms_val:.4f} m" if not np.isnan(rms_val) else ", RMS=N/A"
        ax.set_title(f"({chr(97+i)}) {name} residuals, t={year_label:.2f}{rms_str}")
        cb = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
        cb.set_label("Residual EWH (m)")

    # hide any unused axes
    for j in range(len(products), 4):
        axes.flat[j].set_visible(False)

    fig.suptitle(f"Mascon Fusion: Input Residuals at t={year_label:.2f}", fontsize=13)
    plt.show()
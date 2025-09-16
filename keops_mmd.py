"""
Compute MMD² between PORTAL and INRIX travel times using direct travel time comparison.

This script performs a simple MMD² test on filtered travel time distributions, treating
each observation as a scalar value. NaN values are dropped before computation. This can
and is used with any number of stations/TMCs, as long as there are at least two
observations in each dataset.

Notes:
    Requires CUDA GPU (minimal memory requirements for 1D data).
    Uses PyKeOps for GPU-accelerated kernel computations.
    This is the simplest MMD variant - no feature engineering or missing data handling.

Usage:
    python keops_mmd.py --portal_data data/2023_portal_ts_sjoin_2023_filtered.csv \
                        --inrix_data data/2023_inrix_ts_sjoin_2023_filtered.csv \
                        --n_perms 500 --save True

Required Inputs:
    - PORTAL filtered travel times CSV with columns: stationid (index), portal_tt
    - INRIX filtered travel times CSV with columns: tmc_code (index), inrix_tt

Output:
    If --save=True, creates CSV with filename pattern:
    {portal_file_stem}+{inrix_file_stem}-{column_names}-{n_perms}_perms-{timestamp}.csv
    
    Example: 2023_portal_ts_sjoin_2023_filtered+2023_inrix_ts_sjoin_2023_filtered-portal_tt-inrix_tt-500_perms-240315-1423.csv
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pykeops.torch import LazyTensor
from tqdm import trange

# Argument parsing
parser = argparse.ArgumentParser(
    description="Compute MMD² between PORTAL and INRIX travel times using KeOps"
)

parser.add_argument("--n_perms", type=int, default=1)
parser.add_argument(
    "--portal_data", type=str, default="data/2023_portal_ts_sjoin_2023_filtered.csv"
)
parser.add_argument(
    "--inrix_data", type=str, default="data/2023_inrix_ts_sjoin_2023_filtered.csv"
)
parser.add_argument("--save", type=bool, default=False)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility defaults to 42",
)

args = parser.parse_args()
print(f"Running with args {args}")


# 1.  Data
travel_time_data_files = (
    parser.parse_args().portal_data,
    parser.parse_args().inrix_data,
)

print(f"Loading data from {travel_time_data_files}")


portal_df = pd.read_csv(travel_time_data_files[0], index_col="stationid")
inrix_df = pd.read_csv(travel_time_data_files[1], index_col="tmc_code")

print(f"len(portal_df) before dropna() : {len(portal_df):_}")
portal_df = portal_df.dropna(axis=0, how="any")  # Drop rows with NaN values
print(f"len(portal_df) after dropna() : {len(portal_df):_}")

print(f"len(inrix_df) before dropna() : {len(inrix_df):_}")
inrix_df = inrix_df.dropna(axis=0, how="any")  # Drop rows with NaN values
print(f"len(inrix_df) after dropna() : {len(inrix_df):_}")


print("Contiguous array to gpu")
arr2gpu_start = time.time()
# X == portal, Y == INRIX
portal_cols = [
    "portal_tt",
    # "portal_length_mid", # example additional metadata feature if desired
]
portal_arr = portal_df[portal_cols].to_numpy(dtype=np.float32, copy=False)
portal_arr = np.ascontiguousarray(portal_arr)

X_cpu = torch.from_numpy(portal_arr)
X = X_cpu.pin_memory().to("cuda", non_blocking=True)

print(f"Time to convert portal array to GPU: {time.time() - arr2gpu_start:.2f} seconds")


inrix_cols = [
    "inrix_tt",
    # "Miles" # example additional metadata feature if desired
]
inrix_arr = inrix_df[inrix_cols].to_numpy(dtype=np.float32, copy=False)
inrix_arr = np.ascontiguousarray(inrix_arr)

Y_cpu = torch.from_numpy(inrix_arr)
Y = Y_cpu.pin_memory().to("cuda", non_blocking=True)

print(f"Time to convert INRIX array to GPU: {time.time() - arr2gpu_start:.2f} seconds")

# %%


def mmd_keops(X, Y, gamma=None):
    """Compute the unbiased squared Maximum Mean Discrepancy (MMD^2) between two
    samples using a Gaussian RBF kernel evaluated with KeOps LazyTensors.

    This function avoids forming full MxM, NxN, and MxN kernel matrices in memory
    by computing sums symbolically with KeOps, which is efficient on large datasets
    and supports CPU/GPU tensors.

    Args:
        X (torch.Tensor): A tensor of shape (M, d) containing M samples with d features.
        Y (torch.Tensor): A tensor of shape (N, d) containing N samples with d features.
        gamma (float, optional): Inverse bandwidth of the RBF kernel. If None, uses
            1.0 / d, matching scikit-learn's default for the RBF kernel.

    Returns:
        float: The unbiased estimate of MMD^2 between the empirical distributions
        of X and Y.

    Raises:
        AssertionError: If X and Y do not have the same number of features.

    Notes:
        - Uses the unbiased MMD^2 estimator:
            MMD^2 = E[k(X, X')] + E[k(Y, Y')] - 2 E[k(X, Y)]
          with diagonal self-similarities removed and denominators M(M-1), N(N-1).
        - The RBF kernel is k(u, v) = exp(-gamma * ||u - v||^2).
        - If gamma is None, it defaults to 1/d.
        - Final computation is performed in float64 for improved numerical stability,
          and the returned value is a Python float.
        - Requires M >= 2 and N >= 2 to avoid division by zero in the unbiased estimator.
        - X and Y can reside on CPU or GPU; computations occur on their respective devices.
    """

    assert X.shape[1] == Y.shape[1], "X and Y must have the same number of features"

    M, N, d = X.shape[0], Y.shape[0], X.shape[1]
    # KeOps symbolic variables
    x_i = LazyTensor(X[:, None, :])  # (M, 1, d)  – indexed by i
    x_j = LazyTensor(X[None, :, :])  # (1, M, d)  – indexed by j
    y_i = LazyTensor(Y[:, None, :])  # (N, 1, d)
    y_j = LazyTensor(Y[None, :, :])  # (1, N, d)

    # RBF kernel with gamma = 1 / d   (matches scikit‑learn’s default)
    gam = gamma if gamma is not None else 1.0 / d

    K_xx = (-gam * x_i.sqdist(x_j)).exp()  # (M, M) symbolic block
    K_yy = (-gam * y_i.sqdist(y_j)).exp()  # (N, N)
    K_xy = (-gam * x_i.sqdist(y_j)).exp()  # (M, N)

    # Total sums of the three blocks
    S_xx = K_xx.sum(dim=1).sum()  # torch scalar
    S_yy = K_yy.sum(dim=1).sum()
    S_xy = K_xy.sum(dim=1).sum()

    # Diagonal corrections  (same‑index trick, still O(M d) / O(N d))
    # diag_xx_sum = (-gam * x_i.sqdist(x_i)).exp().sum(dim=1).sum()  # = M for RBF
    # diag_yy_sum = (-gam * y_i.sqdist(y_i)).exp().sum(dim=1).sum()  # = N
    diag_xx_sum = torch.tensor(
        M, dtype=torch.float64, device=X.device
    )  # for RBF kernel
    diag_yy_sum = torch.tensor(
        N, dtype=torch.float64, device=Y.device
    )  # for RBF kernel

    # Unbiased MMD²  (all plain torch scalars from here on)
    M_f = torch.tensor(float(M), dtype=torch.float64, device=X.device)
    N_f = torch.tensor(float(N), dtype=torch.float64, device=Y.device)

    XX = (S_xx.to(torch.float64) - diag_xx_sum) / (M_f * (M_f - 1.0))
    YY = (S_yy.to(torch.float64) - diag_yy_sum) / (N_f * (N_f - 1.0))
    XY = S_xy.to(torch.float64) / (M_f * N_f)

    mmd2 = XX + YY - 2 * XY  # torch scalar

    return mmd2.item()


Z = torch.cat((X, Y))


@torch.no_grad()
def gamma_from_median_heuristic(
    Z: torch.Tensor,
    *,
    max_samples: int = 4000,
    chunk: int = 512,
    g: torch.Generator = None,
) -> float:
    """
    Median heuristic on pooled data Z (n,d).
    Subsamples up to max_samples rows; computes median pairwise distance in chunks.
    Returns gamma = 1 / (2 * sigma^2) with sigma = median distance.
    """
    device = Z.device
    n = Z.shape[0]
    m = min(n, max_samples)

    # sample without replacement (GPU)
    idx = torch.randperm(n, device=device, generator=g)[:m]
    S = Z.index_select(0, idx)  # (m,d)

    meds = []
    for i0 in range(0, m, chunk):
        i1 = min(i0 + chunk, m)
        A = S[i0:i1]  # (a,d)
        for j0 in range(i0, m, chunk):
            j1 = min(j0 + chunk, m)
            B = S[j0:j1]  # (b,d)
            # squared distances (avoid sqrt until the end)
            sq = (A[:, None, :] - B[None, :, :]).pow(2).sum(-1)  # (a,b)
            if i0 == j0:
                tri = torch.triu(sq, diagonal=1)
                vals = tri[tri > 0]
            else:
                vals = sq.reshape(-1)
            if vals.numel():
                meds.append(vals)

    if not meds:
        # degenerate fallback
        return 1e-6

    d2 = torch.cat(meds)  # squared distances
    med = d2.sqrt().median().item()  # median distance (not squared)
    sigma2 = med * med
    return float(1.0 / (2.0 * sigma2))


torch_rng = torch.Generator(device=Z.device).manual_seed(args.seed)

# Compute gamma
print("Start Gamma Compute")
compute_gamma_start = time.time()
median_heuristic_sample_size = 16000

print(
    f"Median heuristic sample size: {median_heuristic_sample_size} (approx {median_heuristic_sample_size / Z.shape[0] * 100:.2f} % of combined dataset size {Z.shape[0]})"
)

median_heuristic_iterations = 100
gammas = torch.tensor(
    [
        gamma_from_median_heuristic(
            Z, max_samples=median_heuristic_sample_size, chunk=1024, g=torch_rng
        )
        for _ in range(median_heuristic_iterations)
    ],
    device=Z.device,
    dtype=Z.dtype,
)
gammas_spread = (torch.max(gammas) - torch.min(gammas)) / torch.mean(gammas)
print(
    f"Gamma spread: {gammas_spread*100:.6g}% with Median Heuristic Iterations: {median_heuristic_iterations}"
)

gamma = torch.median(gammas)  # median of 100 median heuristic gammas

print(
    f"Gamma (median heuristic): {gamma:.6g} (Computed in {time.time() - compute_gamma_start:.2f} seconds)"
)

# %%
print("Start Initial MMD² Compute")
init_mmd2_start = time.time()
initial_mmd2 = mmd_keops(X, Y, gamma=gamma)

print(
    f"Unbiased MMD² : {initial_mmd2} (Computed in {time.time() - init_mmd2_start:.2f} seconds)"
)

# %%
# Permutation test

n_permutations = args.n_perms
N = X.shape[0] + Y.shape[0]

mmd2_values = [initial_mmd2]

start_time = time.time()

for i in trange(n_permutations):
    start_perm = time.time()

    perm = torch.randperm(N, device=Z.device, generator=torch_rng)
    Z_perm = Z[perm]
    X_perm = Z_perm[: X.shape[0]]
    Y_perm = Z_perm[X.shape[0] :]

    perm_mmd2 = mmd_keops(X_perm, Y_perm, gamma=gamma)
    mmd2_values.append(perm_mmd2)


# 8. Save results
if args.save:
    mmd2_values_df = pd.DataFrame(mmd2_values, columns=["mmd2"])

    timestamp_string = datetime.now().strftime("%y%m%d-%H%M")
    mmd2_vals_filename = f"{"+".join([Path(f).stem for f in travel_time_data_files])}-{"-".join(portal_cols+inrix_cols)}-{n_permutations}_perms-{timestamp_string}.csv"
    print(f"Saving MMD² values to {mmd2_vals_filename}")
    mmd2_values_df.to_csv(mmd2_vals_filename, index=False)
else:
    print("Skipping save of MMD² values as --save is False")
# %%

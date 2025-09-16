"""
Compute MMD² between PORTAL and INRIX travel times using matched sensor locations as features.

This script treats each matched PORTAL-INRIX sensor pair as a feature dimension, with time
points as samples. It creates pivoted data from unfiltered time series on the fly, then
computes MMD² using three missing data strategies sequentially.

Notes:
    Requires CUDA GPU (~100GB memory for full year of 15-min data).
    Uses torch for masked computations and PyKeOps for GPU-accelerated kernel computations.

Usage:
    python mmd_spatial_features.py --year 2023 --n_perms 500 --mapping data/2023_portal_inrix_spatial_join_hungarian.csv --save True

Required Inputs:
    - data/{year}_portal_ts_unfiltered.csv: PORTAL time series (stationid, portal_tstamp, portal_tt)
    - data/{year}_inrix_ts_unfiltered.csv: INRIX time series (tmc_code, inrix_tstamp, inrix_tt)
    - Spatial mapping CSV: Maps stationid to tmc_code for matched pairs

Processing Modes (runs all three):
    zfill: Z-score normalize then zero-fill NaN (compare all matched locations)
    mask: Raw values, only compare time points with data at both sensors
    zfill_mask: Z-score but preserve NaN for masking (normalized + missing-aware)

Output:
    If --save=True, creates separate CSV for each mode:
    {year}-spatial_features-{mapping_name}-{mode}-{n_perms}_perms-{timestamp}.csv
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
    description="Compute MMD² between PORTAL and INRIX travel times with matched sensor id as features using KeOps + Torch masked"
)
parser.add_argument(
    "--n_perms",
    type=int,
    default=1,
    help="Number of permutations for the permutation test",
)
parser.add_argument(
    "--mapping",
    type=str,
    default="data/2023_portal_inrix_spatial_join_hungarian.csv",
    help="Path to spatial mapping CSV file",
)
parser.add_argument(
    "--save",
    type=bool,
    default=False,
    help="Whether to save the MMD² results to a file",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility defaults to 42",
)
parser.add_argument(
    "--year", type=int, default=2023, help="Year of the data to process"
)

args = parser.parse_args()

print(f"Running with args {args}")

n_perms = args.n_perms
seed = args.seed
save = args.save
ts_year = args.year
spatial_mapping_file = Path(args.mapping)


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


@torch.no_grad()
def gamma_from_masked_median(
    Z: torch.Tensor,
    M: torch.Tensor,
    *,
    max_samples: int = 4000,
    chunk: int = 512,
    g: torch.Generator = None,
    eps: float = 1e-9,
    rescale_by_d: bool = True,
) -> float:
    """
    Median heuristic on pooled data (Z,M):
    - computes masked pairwise distances (mean over overlapping dims),
    - drops no-overlap pairs,
    - optionally rescales by d so when overlap=d it matches raw sqdist.
    Returns gamma = 1 / (2*sigma^2) with sigma = median distance.
    """
    device = Z.device
    n, d = Z.shape
    m = min(n, max_samples)
    d_t = torch.tensor(float(d), dtype=Z.dtype, device=device)

    # sample without replacement (GPU)
    idx = torch.randperm(n, device=device, generator=g)[:m]
    S = Z.index_select(0, idx)  # (m,d)
    Sm = M.index_select(0, idx)  # (m,d)

    meds = []
    for i0 in range(0, m, chunk):
        i1 = min(i0 + chunk, m)
        A, Am = S[i0:i1], Sm[i0:i1]  # (a,d)
        for j0 in range(i0, m, chunk):
            j1 = min(j0 + chunk, m)
            B, Bm = S[j0:j1], Sm[j0:j1]  # (b,d)

            V = Am[:, None, :] * Bm[None, :, :]  # (a,b,d)
            K = V.sum(-1)  # (a,b) overlap counts
            sq = ((A[:, None, :] - B[None, :, :]).pow(2) * V).sum(-1) / (
                K + eps
            )  # (a,b)
            if rescale_by_d:
                sq = sq * d_t

            valid = K > 0
            if i0 == j0:
                tri = torch.triu(valid, diagonal=1)  # drop diagonal only
                if tri.any():
                    vals = sq[tri]
                    if vals.numel():
                        meds.append(vals[vals > 0])
            else:
                if valid.any():
                    vals = sq[valid]
                    if vals.numel():
                        meds.append(vals[vals > 0])

    if not meds:
        return 1e-6

    d2 = torch.cat(meds)  # squared distances
    med = d2.sqrt().median().item()  # median distance
    sigma2 = med * med if med > 0 else 1e-6
    return float(1.0 / (2.0 * sigma2))


# ----------------------------- Unmasked (Torch) -----------------------------


def _pairwise_sqdist(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """X:(M,d), Y:(N,d) -> D2:(M,N) without (M,N,d) broadcast."""
    X2 = (X * X).sum(dim=1, keepdim=True)  # (M,1)
    Y2 = (Y * Y).sum(dim=1, keepdim=True).T  # (1,N)
    D2 = X2 + Y2 - 2.0 * (X @ Y.T)
    return D2.clamp_min(0)


def mmd_torch(X: torch.Tensor, Y: torch.Tensor, gamma: float | None = None) -> float:
    """Unbiased MMD^2 with RBF kernel (unmasked)."""
    assert X.shape[1] == Y.shape[1], "feature dims must match"
    M, N, d = X.shape[0], Y.shape[0], X.shape[1]
    gam = gamma if gamma is not None else 1.0 / d

    D2_xx = _pairwise_sqdist(X, X)
    D2_yy = _pairwise_sqdist(Y, Y)
    D2_xy = _pairwise_sqdist(X, Y)

    K_xx = torch.exp(-gam * D2_xx)
    K_yy = torch.exp(-gam * D2_yy)
    K_xy = torch.exp(-gam * D2_xy)

    S_xx = K_xx.sum()
    S_yy = K_yy.sum()
    S_xy = K_xy.sum()

    diag_xx = torch.tensor(M, dtype=torch.float64, device=X.device)
    diag_yy = torch.tensor(N, dtype=torch.float64, device=Y.device)

    M_f = torch.tensor(float(M), dtype=torch.float64, device=X.device)
    N_f = torch.tensor(float(N), dtype=torch.float64, device=Y.device)

    XX = (S_xx.to(torch.float64) - diag_xx) / (M_f * (M_f - 1.0))
    YY = (S_yy.to(torch.float64) - diag_yy) / (N_f * (N_f - 1.0))
    XY = S_xy.to(torch.float64) / (M_f * N_f)

    return float((XX + YY - 2.0 * XY).clamp_min(0))


# -------------------------- Masked distance (Torch) -------------------------


def _masked_pairwise_sqdist_mean(
    X: torch.Tensor,
    Mx: torch.Tensor,
    Y: torch.Tensor,
    My: torch.Tensor,
    *,
    eps: float = 1e-9,
    rescale_by_d: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Masked *mean* squared distance per pair using matmuls only.
    Returns:
      D2:(M,N)  masked mean of (x - y)^2 over overlapping features
      V :(M,N)  0/1 validity (1 if any overlap, else 0)
    """
    assert X.shape == Mx.shape and Y.shape == My.shape
    assert X.shape[1] == Y.shape[1], "feature dims must match"
    d = X.shape[1]
    scale = float(d) if rescale_by_d else 1.0

    # Overlap counts k_ij = sum_t m_i,t * m_j,t
    k = Mx @ My.T  # (M,N)

    # Numerator: sum_t m_i m_j (x^2 + y^2 - 2xy) via matmuls
    Sx2 = Mx * (X * X)  # (M,d)
    Sy2 = My * (Y * Y)  # (N,d)
    SX = Mx * X  # (M,d)
    SY = My * Y  # (N,d)

    num = (Sx2 @ My.T) + (Mx @ Sy2.T) - 2.0 * (SX @ SY.T)  # (M,N)
    D2 = (num / (k + eps)).clamp_min(0) * scale  # (M,N)

    V = (k > 0).to(X.dtype)  # (M,N) in {0,1}
    return D2, V


def mmd_torch_masked(
    X: torch.Tensor,
    Mx: torch.Tensor,
    Y: torch.Tensor,
    My: torch.Tensor,
    *,
    gamma: float | None = None,
    eps: float = 1e-9,
    rescale_by_d: bool = True,
    use_valid_pair_denominators: bool = True,
) -> float:
    """
    Unbiased masked MMD^2 with RBF kernel (pure Torch).
    - Mask inside the sum (mean over overlaps), no (M,N,d) tensors.
    - Invalid pairs have kernel=0 and are excluded via denominators.
    """
    assert X.shape[1] == Y.shape[1], "feature dims must match"
    assert X.shape == Mx.shape and Y.shape == My.shape, "data/mask shapes must match"
    M, N, d = X.shape[0], Y.shape[0], X.shape[1]
    gam = gamma if gamma is not None else 1.0 / d

    D2_xx, V_xx = _masked_pairwise_sqdist_mean(
        X, Mx, X, Mx, eps=eps, rescale_by_d=rescale_by_d
    )
    D2_yy, V_yy = _masked_pairwise_sqdist_mean(
        Y, My, Y, My, eps=eps, rescale_by_d=rescale_by_d
    )
    D2_xy, V_xy = _masked_pairwise_sqdist_mean(
        X, Mx, Y, My, eps=eps, rescale_by_d=rescale_by_d
    )

    K_xx = torch.exp(-gam * D2_xx) * V_xx
    K_yy = torch.exp(-gam * D2_yy) * V_yy
    K_xy = torch.exp(-gam * D2_xy) * V_xy

    S_xx_all = K_xx.sum()
    S_yy_all = K_yy.sum()
    S_xy = K_xy.sum()

    # Diagonal corrections: rows with ≥1 observed feature have K(ii)=1
    diag_xx = (Mx.sum(dim=1) > 0).to(torch.float64).sum().to(torch.float64)
    diag_yy = (My.sum(dim=1) > 0).to(torch.float64).sum().to(torch.float64)

    if use_valid_pair_denominators:
        Den_xx = (V_xx.sum().to(torch.float64) - diag_xx).clamp_min(1.0)
        Den_yy = (V_yy.sum().to(torch.float64) - diag_yy).clamp_min(1.0)
        Den_xy = V_xy.sum().to(torch.float64).clamp_min(1.0)
    else:
        Den_xx = torch.tensor(float(M * (M - 1)), dtype=torch.float64, device=X.device)
        Den_yy = torch.tensor(float(N * (N - 1)), dtype=torch.float64, device=Y.device)
        Den_xy = torch.tensor(float(M * N), dtype=torch.float64, device=X.device)

    XX = (S_xx_all.to(torch.float64) - diag_xx) / Den_xx
    YY = (S_yy_all.to(torch.float64) - diag_yy) / Den_yy
    XY = S_xy.to(torch.float64) / Den_xy

    return float((XX + YY - 2.0 * XY).clamp_min(0))


# --------------------------- Median-gamma (Torch) -------------------------------


@torch.no_grad()
def gamma_from_median_heuristic_torch(
    Z: torch.Tensor, *, max_samples: int = 4000, g: torch.Generator | None = None
) -> float:
    """Unmasked median heuristic (same as your KeOps gamma, but Torch)."""
    n = Z.shape[0]
    m = min(n, max_samples)
    idx = torch.randperm(n, device=Z.device, generator=g)[:m]
    S = Z.index_select(0, idx)  # (m,d)
    D2 = _pairwise_sqdist(S, S)  # (m,m)
    iu = torch.triu(torch.ones(m, m, dtype=torch.bool, device=Z.device), diagonal=1)
    vals = D2[iu]
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return 1e-6
    med = vals.sqrt().median().item()
    sigma2 = med * med if med > 0 else 1e-6
    return float(1.0 / (2.0 * sigma2))


@torch.no_grad()
def gamma_from_masked_median_torch(
    Z: torch.Tensor,
    M: torch.Tensor,
    *,
    max_samples: int = 4000,
    g: torch.Generator | None = None,
    eps: float = 1e-9,
    rescale_by_d: bool = True,
) -> float:
    """
    Masked median heuristic using the SAME masked distance as mmd_torch_masked.
    """
    assert Z.shape == M.shape
    n, d = Z.shape
    m = min(n, max_samples)
    idx = torch.randperm(n, device=Z.device, generator=g)[:m]
    S, Sm = Z.index_select(0, idx), M.index_select(0, idx)  # (m,d), (m,d)

    D2, V = _masked_pairwise_sqdist_mean(
        S, Sm, S, Sm, eps=eps, rescale_by_d=rescale_by_d
    )

    iu = torch.triu(torch.ones(m, m, dtype=torch.bool, device=Z.device), diagonal=1)
    keep = iu & (V > 0)
    vals = D2[keep]
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return 1e-6
    med = vals.sqrt().median().item()
    sigma2 = med * med if med > 0 else 1e-6
    return float(1.0 / (2.0 * sigma2))


# Processing helpers
def zscore_fill(df):
    """Z-score each column using observed values, then fill NaN -> 0 (neutral)."""
    mu = df.mean(axis=0, skipna=True)
    sd = df.std(axis=0, skipna=True).replace(0, 1.0)
    return ((df - mu) / sd).fillna(0.0)


def zscore_keepna(df):
    """Z-score each column but KEEP NaN (for masking later)."""
    mu = df.mean(axis=0, skipna=True)
    sd = df.std(axis=0, skipna=True).replace(0, 1.0)
    return (df - mu) / sd  # don't fill


def to_tensor(df, device="cuda", dtype=np.float64):
    """Just make a tensor from contiguous array."""
    arr = np.ascontiguousarray(df.to_numpy(dtype=dtype, copy=False).astype(np.float64))
    t = torch.from_numpy(arr).to(device)
    return t


def to_tensor_and_mask(df, device="cuda", dtype=np.float64):
    """
    Returns:
      X : torch.(n,d) numeric tensor with NaNs replaced by 0 (safe filler)
      M : torch.(n,d) float mask in {0,1} where 1 = observed, 0 = missing
    """
    arr = df.to_numpy(dtype=dtype, copy=False).astype(np.float64)
    mask = ~np.isnan(arr)
    arr_filled = np.nan_to_num(arr, nan=0.0)
    X = torch.from_numpy(np.ascontiguousarray(arr_filled)).to(device)
    M = torch.from_numpy(np.ascontiguousarray(mask.astype(arr_filled.dtype))).to(device)
    return X, M


# Set up torch RNG for reproducibility
torch_rng = torch.Generator(device="cuda").manual_seed(seed)


freq = "15min"
time_grid = pd.date_range(
    pd.Timestamp(ts_year, 1, 1, tz="America/Los_Angeles"),
    pd.Timestamp(ts_year + 1, 1, 1, tz="America/Los_Angeles"),
    inclusive="left",
    freq=freq,
)

start = time.time()
# Unfiltered PORTAL (whole system)
print("Reading unfiltered data...")
print(f"Data year: {ts_year}")
print("portal...")
portal_ts_df_unfiltered = pd.read_csv(f"data/{ts_year}_portal_ts_unfiltered.csv")
portal_ts_df_unfiltered = portal_ts_df_unfiltered.convert_dtypes().assign(
    stationid=lambda df: df["stationid"].astype("string"),
    portal_tstamp=lambda df: pd.to_datetime(
        df["portal_tstamp"], utc=True, errors="coerce"
    ).dt.tz_convert("America/Los_Angeles"),
    portal_tt=lambda df: pd.to_numeric(df["portal_tt"], errors="coerce"),
)
# Unfiltered INRIX (whole system)
print("inrix...")
inrix_ts_df_unfiltered = pd.read_csv(f"data/{ts_year}_inrix_ts_unfiltered.csv")

inrix_ts_df_unfiltered = inrix_ts_df_unfiltered.convert_dtypes().assign(
    tmc_code=lambda df: df["tmc_code"].astype("string"),
    inrix_tstamp=lambda df: pd.to_datetime(
        df["inrix_tstamp"], utc=True, errors="coerce"
    ).dt.tz_convert("America/Los_Angeles"),
    inrix_tt=lambda df: pd.to_numeric(df["inrix_tt"], errors="coerce"),
)

print(f"Time to read unfiltered data: {time.time() - start:.2f} seconds")

print("Load spatial mappings...")
spatial_mapping_df = pd.read_csv(spatial_mapping_file)
spatial_mapping_df = spatial_mapping_df.assign(
    stationid=lambda df: df["stationid"].astype("string"),
    tmc_code=lambda df: df["tmc"].astype("string"),
)

tmc_order = spatial_mapping_df["tmc_code"].tolist()
stationid_order = spatial_mapping_df["stationid"].tolist()


print("Creating unfiltered pivots...")
print("portal...")
start_pivot = time.time()

P_spatial_features = (
    portal_ts_df_unfiltered.pivot(
        index="portal_tstamp", columns="stationid", values="portal_tt"
    )
    .reindex(index=time_grid)  # align to full-year grid
    .reindex(columns=stationid_order)  # enforce spatial mapping order
    .sort_index(axis=0)
    .astype(np.float64)
)
print(f"Time to create portal pivot: {time.time() - start_pivot:.2f} seconds")


print("inrix...")
I_spatial_features = (
    inrix_ts_df_unfiltered.pivot(
        index="inrix_tstamp", columns="tmc_code", values="inrix_tt"
    )
    .reindex(index=time_grid)  # align to full-year grid
    .reindex(columns=tmc_order)  # enforce spatial mapping order
    .sort_index(axis=0)
    .astype(np.float64)
)
print(f"Time to create inrix pivot: {time.time() - start_pivot:.2f} seconds")

portal_df = P_spatial_features
inrix_df = I_spatial_features

print(f"len(portal_df) : {len(portal_df):_} x {len(portal_df.columns):_} features")
print(f"len(inrix_df) : {len(inrix_df):_} x {len(inrix_df.columns):_} features")

for mode in ["zfill", "mask", "zfill_mask"]:
    print(f"Mode: {mode}")
    if mode == "zfill":
        # Scale + single-step missing handling (NaN -> 0 after z-score)
        X = to_tensor(zscore_fill(portal_df))
        Y = to_tensor(zscore_fill(inrix_df))
        Z = torch.cat([X, Y], 0)
        print("Computing unmasked median heuristic gamma... (zfill)")
        compute_gamma_start = time.time()
        median_heuristic_sample_size = min(4000, Z.shape[0])
        print(
            f"Median heuristic sample size: {median_heuristic_sample_size} (approx {median_heuristic_sample_size / Z.shape[0] * 100:.2f} % of combined dataset size {Z.shape[0]})"
        )
        median_heuristic_iterations = 100
        # unmasked median heuristic
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
        print(
            f"Gamma spread: {((torch.max(gammas) - torch.min(gammas)) / torch.mean(gammas))*100:.6g}% with Median Heuristic Iterations: {median_heuristic_iterations}"
        )
        gamma = torch.median(gammas)  # median of 100 median heuristic gammas
        print(
            f"Gamma (zfill median heuristic): {gamma:.6g} (Computed in {time.time() - compute_gamma_start:.2f} seconds)"
        )
        print("Start Initial MMD² Compute (zfill)")
        init_mmd2_start = time.time()
        initial_mmd2 = mmd_keops(X, Y, gamma=gamma)

        print(
            f"Initial Unbiased MMD² (zfill): {initial_mmd2} (Computed in {time.time() - init_mmd2_start:.2f} seconds)"
        )

        n_permutations = n_perms
        N = X.shape[0] + Y.shape[0]

        mmd2_values = [initial_mmd2]

        for i in trange(n_permutations):
            perm = torch.randperm(N, device=Z.device, generator=torch_rng)
            Z_perm = Z[perm]
            X_perm = Z_perm[: X.shape[0]]
            Y_perm = Z_perm[X.shape[0] :]

            perm_mmd2 = mmd_keops(X_perm, Y_perm, gamma=gamma)
            mmd2_values.append(perm_mmd2)

    elif mode == "mask":

        # No scaling; compare only overlapping slots (missing ignored)
        X, Mx = to_tensor_and_mask(portal_df, dtype=np.float32)
        Y, My = to_tensor_and_mask(inrix_df, dtype=np.float32)
        Z, Mz = torch.cat([X, Y], 0), torch.cat([Mx, My], 0)

        print("Computing masked median heuristic gamma... (masked)")
        compute_gamma_start = time.time()
        median_heuristic_sample_size = min(4000, Z.shape[0])
        print(
            f"Median heuristic sample size: {median_heuristic_sample_size} (approx {median_heuristic_sample_size / Z.shape[0] * 100:.2f} % of combined dataset size {Z.shape[0]})"
        )
        median_heuristic_iterations = 100
        gammas = torch.tensor(
            [
                gamma_from_masked_median_torch(
                    Z,
                    Mz,
                    max_samples=median_heuristic_sample_size,
                    g=torch_rng,
                    rescale_by_d=True,
                )
                for _ in range(median_heuristic_iterations)
            ],
            device=Z.device,
            dtype=Z.dtype,
        )

        print(
            f"Gamma spread: {((torch.max(gammas) - torch.min(gammas)) / torch.mean(gammas))*100:.6g}% with Median Heuristic Iterations: {median_heuristic_iterations}"
        )
        gamma = torch.median(gammas)  # median of 100 masked median heuristic gammas
        print(
            f"Gamma (masked median heuristic): {gamma:.6g} (Computed in {time.time() - compute_gamma_start:.2f} seconds)"
        )
        print("Start Initial MMD² Compute (masked)")
        init_mmd2_start = time.time()
        initial_mmd2 = mmd_torch_masked(
            X,
            Mx,
            Y,
            My,
            gamma=gamma,
            rescale_by_d=True,
            use_valid_pair_denominators=True,
        )

        print(
            f"Initial Unbiased MMD² (masked): {initial_mmd2} (Computed in {time.time() - init_mmd2_start:.2f} seconds)"
        )
        n_permutations = n_perms
        N = X.shape[0] + Y.shape[0]

        mmd2_values = [initial_mmd2]

        for i in trange(n_permutations):
            perm = torch.randperm(N, device=Z.device, generator=torch_rng)
            Z_perm, M_perm = Z[perm], Mz[perm]
            X_perm, Mx_perm = Z_perm[: X.shape[0]], M_perm[: X.shape[0]]
            Y_perm, My_perm = Z_perm[X.shape[0] :], M_perm[X.shape[0] :]

            perm_mmd2 = mmd_torch_masked(
                X_perm,
                Mx_perm,
                Y_perm,
                My_perm,
                gamma=gamma,
                rescale_by_d=True,
                use_valid_pair_denominators=True,
            )
            mmd2_values.append(perm_mmd2)

    elif mode == "zfill_mask":
        # Scale but KEEP NaNs for masking; then mask handles remaining gaps
        X, Mx = to_tensor_and_mask(zscore_keepna(portal_df))
        Y, My = to_tensor_and_mask(zscore_keepna(inrix_df))
        Z, Mz = torch.cat([X, Y], 0), torch.cat([Mx, My], 0)

        print("Computing masked median heuristic gamma... (zfill_mask)")
        compute_gamma_start = time.time()
        median_heuristic_sample_size = min(4000, Z.shape[0])
        print(
            f"Median heuristic sample size: {median_heuristic_sample_size} (approx {median_heuristic_sample_size / Z.shape[0] * 100:.2f} % of combined dataset size {Z.shape[0]})"
        )
        median_heuristic_iterations = 100
        gammas = torch.tensor(
            [
                gamma_from_masked_median_torch(
                    Z,
                    Mz,
                    max_samples=median_heuristic_sample_size,
                    g=torch_rng,
                    rescale_by_d=True,
                )
                for _ in range(median_heuristic_iterations)
            ],
            device=Z.device,
            dtype=Z.dtype,
        )
        print(
            f"Gamma spread: {((torch.max(gammas) - torch.min(gammas)) / torch.mean(gammas))*100:.6g}% with Median Heuristic Iterations: {median_heuristic_iterations}"
        )
        gamma = torch.median(gammas)  # median of 100 masked median heuristic gammas
        print(
            f"Gamma (zfill_mask median heuristic): {gamma:.6g} (Computed in {time.time() - compute_gamma_start:.2f} seconds)"
        )
        print("Start Initial MMD² Compute (zfill_mask)")
        init_mmd2_start = time.time()
        initial_mmd2 = mmd_torch_masked(
            X,
            Mx,
            Y,
            My,
            gamma=gamma,
            rescale_by_d=True,
            use_valid_pair_denominators=True,
        )
        print(
            f"Initial MMD² (zfill_mask): {initial_mmd2:.6g} (Computed in {time.time() - init_mmd2_start:.2f} seconds)"
        )

        n_permutations = n_perms
        N = X.shape[0] + Y.shape[0]

        mmd2_values = [initial_mmd2]

        for i in trange(n_permutations):
            perm = torch.randperm(N, device=Z.device, generator=torch_rng)
            Z_perm, M_perm = Z[perm], Mz[perm]
            X_perm, Mx_perm = Z_perm[: X.shape[0]], M_perm[: X.shape[0]]
            Y_perm, My_perm = Z_perm[X.shape[0] :], M_perm[X.shape[0] :]

            perm_mmd2 = mmd_torch_masked(
                X_perm,
                Mx_perm,
                Y_perm,
                My_perm,
                gamma=gamma,
                rescale_by_d=True,
                use_valid_pair_denominators=True,
            )
            mmd2_values.append(perm_mmd2)

    if save:
        mmd2_values_df = pd.DataFrame(mmd2_values, columns=["mmd2"])

        timestamp_string = datetime.now().strftime("%y%m%d-%H%M")
        mmd2_vals_filename = f"{ts_year}-spatial_features-{spatial_mapping_file.stem}-{mode}-{n_permutations}_perms-{timestamp_string}.csv"
        print(f"Saving MMD² values to {mmd2_vals_filename}")
        mmd2_values_df.to_csv(mmd2_vals_filename, index=False)
    else:
        print("Skipping save of MMD² values as --save is False")

# %%

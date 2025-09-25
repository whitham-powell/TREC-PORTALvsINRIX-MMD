# TREC: PORTAL vs INRIX MMD Analysis

Statistical comparison of traffic travel time distributions between PORTAL (traffic detector) and INRIX (GPS probe) data sources using Maximum Mean Discrepancy (MMD) testing.

## Overview

This repository implements MMD² statistical testing to quantify distributional differences between PORTAL detector-based travel times and INRIX probe-based travel times. The analysis handles the challenge of comparing fundamentally different traffic data sources: point-based detector measurements vs. segment-based GPS traces.

The codebase provides multiple MMD computation strategies to handle missing data patterns and different feature representations, enabling robust statistical comparison despite sparse and misaligned observations.

## Key Features

- **Spatial matching**: Hungarian algorithm-based optimal pairing of PORTAL stations to INRIX TMC segments
- **Missing data strategies**: Zero-fill, masking, and hybrid approaches for handling sparse observations
- **GPU acceleration**: PyKeOps and PyTorch implementations for large-scale computation
- **Multiple MMD variants**: Direct comparison, spatial features view, temporal features view
- **Flexible feature engineering**: Support for both sensor-as-feature and time-as-feature representations

## Requirements

- Python 3.12+
- CUDA-capable GPU (required)
  - Minimum: ~4GB for filtered data
  - Recommended: 100GB for full unfiltered analysis with masking
- Key dependencies:
  - PyTorch with CUDA support
  - PyKeOps for GPU-accelerated kernel computations
  - GeoPandas for spatial operations
  - SciPy for optimization algorithms

## Installation

This project includes `pyproject.toml` and `uv.lock` files for reproducible environment management. The project was developed and tested using [uv](https://github.com/astral-sh/uv) for dependency management.

### Recommended: Using uv (tested approach)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/yourusername/TREC-PORTALvsINRIX-MMD.git
cd TREC-PORTALvsINRIX-MMD

# Create environment and install dependencies from lock file
uv sync

# Run scripts with uv
uv run python keops_mmd.py --help
```

### Alternative: Manual installation

If uv is not available, you can manually install dependencies, though exact package versions may differ from the tested configuration:

```bash
# Clone repository
git clone https://github.com/yourusername/TREC-PORTALvsINRIX-MMD.git
cd TREC-PORTALvsINRIX-MMD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (versions may differ from lock file)
pip install torch pykeops geopandas scipy pandas numpy shapely tqdm
```

**Note**: This project was developed and tested exclusively using uv. For reproducible results matching our experiments, we strongly recommend using the uv installation method with the provided lock file.

## Data Preparation

### Spatial Matching

Pre-computed PORTAL-INRIX spatial mappings are provided in the `spatial_mappings/` directory. The MMD computation scripts use these mappings directly.

The `create_portal_inrix_sjoin.py` script is included for reference and reproducibility, but note that it:

- Requires preprocessed INRIX shapefiles converted to Parquet format
- Makes specific assumptions about data structure and geometry formats
- Was not the primary workflow for our experiments

If you need to generate your own spatial mappings:

```bash
# Using uv (recommended)
uv run python create_portal_inrix_sjoin.py \
    --year 2023 \
    --max_dist 10.0 \
    --threshold 0.5

# Expected inputs (must be prepared separately):
# - PORTAL stations/highways CSVs with WKB geometry columns
# - INRIX TMC geometries in Parquet format (preprocessed from shapefiles)
```

**Output:**

- `data/{year}_portal_inrix_spatial_join_hungarian.csv` - Full metadata for matched pairs
- `data/{year}_portal_inrix_spatial_join_hungarian_ids_only.csv` - Minimal ID mapping

## MMD Computation Scripts

### Direct Comparison (keops_mmd.py)

Compare travel times directly as scalar distributions. Each observation is treated as an individual sample.

```bash
uv run python keops_mmd.py \
    --portal_data data/2023_portal_ts_sjoin_2023_filtered.csv \
    --inrix_data data/2023_inrix_ts_sjoin_2023_filtered.csv \
    --n_perms 500 \
    --save True
```

**Note**: This approach becomes computationally intensive with full timeseries data (millions of samples).

### Spatial Features View (mmd_spatial_features.py)

Treat matched sensor pairs as features, time points as samples (~35k samples × ~200 features):

```bash
uv run python mmd_spatial_features.py \
    --year 2023 \
    --n_perms 500 \
    --mapping spatial_mappings/2023_portal_inrix_spatial_join_hungarian.csv \
    --save True
```

Processes three missing data strategies sequentially:

1. **zfill**: Replace NaN with 0 after z-score normalization
2. **mask**: Use only overlapping observations
3. **zfill_mask**: Hybrid approach combining both strategies

### Temporal Features View

Treat time slots as features, sensors as samples (~200 samples × ~35k features):

```bash
# Pre-pivoted filtered data (mmd_time_features.py)
uv run python mmd_time_features.py \
    --portal_data data/2023_portal_15min_pivot_filtered.csv \
    --inrix_data data/2023_inrix_15min_pivot_filtered.csv \
    --mode zfill \
    --n_perms 500 \
    --save True

# Unfiltered data - creates pivots on-the-fly (unfiltered_mmd_time_features.py)
uv run python unfiltered_mmd_time_features.py \
    --year 2023 \
    --n_perms 500 \
    --save True
```

## Performance Considerations

| Script                          | Data Shape                                        | GPU Memory                   | Runtime (500 perms)                |
| ------------------------------- | ------------------------------------------------- | ---------------------------- | ---------------------------------- |
| keops_mmd.py                    | Filtered (~thousands of samples × 1 feature)      | <1GB                         | ~5 minutes                         |
| keops_mmd.py                    | Full timeseries (millions of samples × 1 feature) | ~4GB                         | 12 hours (3090) / 2.5 hours (H200) |
| mmd_spatial_features.py         | ~35k samples × ~200 features                      | ~100GB                       | 2-3 hours                          |
| mmd_time_features.py            | ~200 samples × ~35k features                      | ~4GB (zfill) / ~100GB (mask) | 30-60 minutes                      |
| unfiltered_mmd_time_features.py | ~200 samples × ~35k features                      | ~4GB (zfill) / ~100GB (mask) | 1-2 hours                          |

> Note: GPU memory usage varies significantly based on the missing data strategy employed and are approximate values.

**Key insight**: The time features approach is more computationally efficient than direct comparison despite high dimensionality, as it reshapes millions of observations into a more tractable matrix structure.

## Output Files

All scripts generate CSV files with MMD² values (initial + permutation tests):

- Row 0: Observed MMD² between PORTAL and INRIX
- Rows 1-n: MMD² values from permutation tests
- Filename pattern: `{data_descriptors}-{columns}-{n_perms}_perms-{timestamp}.csv`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{trec_portal_inrix_mmd,
  title = {{TREC-PORTAL vs INRIX MMD Analysis: Statistical Comparison of Traffic Data Sources}},
  author = {Elijah Whitham-Powell},
  year = {2025},
  url = {https://github.com/whitham-powell/TREC-PORTALvsINRIX-MMD},
  note = {Software for computing Maximum Mean Discrepancy between PORTAL and INRIX traffic data}
}
```

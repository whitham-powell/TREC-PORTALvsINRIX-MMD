# create_portal_inrix_sjoin.py - used to create the spatial join between INRIX and PORTAL data
# using GeoPandas and the Hungarian algorithm.
# The script reads in metadata and geometries for both datasets, standardizes directions,
# performs a spatial join to find nearest matches, and applies the Hungarian algorithm to
# optimize the matching based on distance and overlap length.
# The final matched pairs are saved to CSV files for further analysis.

# %%
import argparse
import struct

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.pylab import f
from scipy.optimize import linear_sum_assignment
from shapely.wkb import loads as wkb_loads


parser = argparse.ArgumentParser(
    description="Create a spatial join between INRIX and PORTAL data using GeoPandas and the Hungarian algorithm."
)
parser.add_argument(
    "--p_meta_highways",
    type=str,
    default="data/portal_highways.csv",
    help="Path to PORTAL highways metadata CSV used to join PORTAL stations with highway names and directions.",
)
parser.add_argument(
    "--p_meta_stations",
    type=str,
    default="data/portal_stations.csv",
    help="Path to PORTAL stations metadata CSV used to join PORTAL stations with highway names and directions.",
)
parser.add_argument(
    "--i_meta",
    type=str,
    default="data/inrix_metadata.csv",
    help="Path to INRIX metadata CSV used to narrow down the INRIX data to the relevant TMC codes.",
)
parser.add_argument(
    "--i_geoms",
    type=str,
    default="data/processed/inrix_tmc/inrix_tmc_2023.parquet",
    help="Path to processed INRIX TMC geometries Parquet file.",
)
parser.add_argument(
    "--year",
    type=str,
    default="2023",
    help="Year of the data to process (e.g., 2019, 2020, 2021, 2022, 2023). Default is 2023.",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Threshold considered a tie for the Hungarian algorithm to match INRIX and PORTAL stations. Default is 0.5. (Units: meters)",
)
parser.add_argument(
    "--max_dist",
    type=float,
    default=10.0,
    help="Maximum distance for sjoin_nearest(). Default is 10 meters.",
)

args = parser.parse_args()
print(f"Running with args {args}")


def standardize_direction(direction, bound=None):
    """
    Standardize direction formats between PORTAL and INRIX
    """
    direction_map = {
        # INRIX formats
        "NORTHBOUND": "NORTH",
        "SOUTHBOUND": "SOUTH",
        "WESTBOUND": "WEST",
        "EASTBOUND": "EAST",
        # PORTAL formats
        "NORTH": "NORTH",
        "SOUTH": "SOUTH",
        "WEST": "WEST",
        "EAST": "EAST",
        "NORT": "NORTH",  # Handle the truncated 'NORT'
        "CONST": None,  # Handle construction case separately
    }

    # If direction is not valid, try to use bound
    bound_map = {
        "NB": "NORTH",
        "SB": "SOUTH",
        "WB": "WEST",
        "EB": "EAST",
        "JB": None,  # Special cases
        "ZB": None,
    }

    std_direction = direction_map.get(direction.upper())
    if std_direction is None and bound is not None:
        std_direction = bound_map.get(bound.upper())

    return std_direction


def check_ewkb_srid(wkb_hex):
    # Convert hex to bytes
    wkb_bytes = bytes.fromhex(wkb_hex)

    # Check endianness (byte order)
    endian = ">" if wkb_bytes[0] == 0 else "<"

    # Check if it's EWKB (has SRID)
    has_srid = bool(struct.unpack(endian + "I", wkb_bytes[1:5])[0] & 0x20000000)

    if has_srid:
        # SRID is stored after the type indicator
        srid = struct.unpack(endian + "I", wkb_bytes[5:9])[0]
        return srid
    return None


def wkb_to_geom(wkb_hex):
    try:
        if pd.isna(wkb_hex) or not wkb_hex.strip():
            return None
        geom = wkb_loads(bytes.fromhex(wkb_hex))
        # print(f"ekwb_srid: {check_ewkb_srid(wkb_hex)}")  # Uncomment to check SRID
        return geom
    except (ValueError, TypeError) as e:
        print(f"Conversion error: {e}")
        return None


def tag(df, tag, keys):
    return (
        df.set_index(keys)
        .add_prefix(f"{tag}_")
        .rename_axis(keys)  # put key names back
        .reset_index()
    )


def hungarian_dist(cand_gdf, threshold=0.5):
    eps = threshold  # meters threshold for "same" distance
    C = cand_gdf[["stationid", "tmc", "gpd_distance", "overlap_len"]].copy()
    C["dist_scaled"] = np.log1p(C["gpd_distance"])

    # Per (station,TMC) keep smallest distance; within eps, keep largest overlap
    C = C.sort_values(
        ["stationid", "tmc", "dist_scaled", "overlap_len"],
        ascending=[True, True, True, False],
    )

    # group by (station,TMC) and squash near-duplicates
    def squash(g):
        g = g.copy()
        g["d0"] = g["dist_scaled"].min()
        return g[g["dist_scaled"] <= g["d0"] + np.log1p(eps)].nlargest(1, "overlap_len")

    C_edge = (
        C.groupby(["stationid", "tmc"], group_keys=False)[
            ["dist_scaled", "overlap_len"]
        ]
        .apply(squash)
        .reset_index(drop=True)
    )

    # keep ONE row per (stationid, Tmc): the smallest distance
    idx = C.groupby(["stationid", "tmc"])["dist_scaled"].idxmin()
    C_edge = C.loc[idx].reset_index(drop=True)

    # build cost matrix
    stations = C_edge["stationid"].unique()
    tmcs = C_edge["tmc"].unique()
    s2i = {s: i for i, s in enumerate(stations)}
    t2j = {t: j for j, t in enumerate(tmcs)}

    BIG = 1e12
    M = np.full((len(stations), len(tmcs)), BIG, float)
    for _, r in C_edge.iterrows():
        i, j = s2i[r["stationid"]], t2j[r["tmc"]]
        M[i, j] = min(M[i, j], float(r["dist_scaled"]))

    # solve
    ri, cj = linear_sum_assignment(M)

    # pairs chosen
    res = pd.DataFrame(
        [
            {"stationid": stations[i], "tmc": tmcs[j]}
            for i, j in zip(ri, cj)
            if M[i, j] < BIG
        ]
    )

    # merge back details
    hungarian_dist = res.merge(
        C_edge, on=["stationid", "tmc"], how="left", validate="one_to_one"
    )

    # sanity
    assert hungarian_dist["stationid"].is_unique
    assert hungarian_dist["tmc"].is_unique

    return hungarian_dist


# script params

year_str = args.year
portal_stations_meta_file = args.p_meta_stations
portal_highways_meta_file = args.p_meta_highways
inrix_meta_file = args.i_meta
inrix_geoms_file = args.i_geoms
hung_threshold = args.threshold
nearest_distance_limit = args.max_dist

portal_meta_cols = [
    "stationid",
    "portal_highwayname",
    "portal_direction",
    "portal_seg_len_m",
]
inrix_meta_cols = [
    "tmc",
    "inrix_route_id",
    "inrix_standardized_direction",
    "inrix_seg_len_m",
]

# Load CSV

portal_stations_df = pd.read_csv(portal_stations_meta_file)
portal_highways_df = pd.read_csv(portal_highways_meta_file)

# Convert hex WKB to geometries
portal_stations_df["segment_geom"] = portal_stations_df["segment_geom"].apply(
    wkb_to_geom
)
portal_stations_df["station_geom"] = portal_stations_df["station_geom"].apply(
    wkb_to_geom
)

# After your wkb_to_geom conversions
print(f"Total rows in DataFrame: {len(portal_stations_df)}")
print(
    f"Successful segment conversions: {portal_stations_df['segment_geom'].notna().sum()}"
)
print(
    f"Successful station conversions: {portal_stations_df['station_geom'].notna().sum()}"
)

# Merge highways data for additional station direction coverage and highway names
portal_stations_df = portal_stations_df.merge(
    portal_highways_df[["highwayid", "direction", "bound", "highwayname"]],
    on="highwayid",
    how="left",
).assign(
    highwayid=lambda df: df["highwayid"].astype("Int64"),
    stationid=lambda df: df["stationid"].astype("Int64").astype(str),
)

portal_stations_df["standardized_direction"] = portal_stations_df.apply(
    lambda row: standardize_direction(row["direction"], row["bound"]),
    axis=1,
)

portal_stations_df = tag(portal_stations_df, "portal", ["stationid"])


# Convert to GeoDataFrame with Web Mercator CRS
the_CRS = "EPSG:3857"
portal_gdf = gpd.GeoDataFrame(
    portal_stations_df, geometry="portal_segment_geom", crs=the_CRS
).assign(portal_seg_len_m=lambda df: df.geometry.length)


tmc_identification_df = pd.read_csv(inrix_meta_file)
inrix_full_gdf = gpd.read_parquet(inrix_geoms_file).to_crs(the_CRS)
inrix_full_gdf["standardized_direction"] = inrix_full_gdf.apply(
    lambda row: standardize_direction(row["direction"]), axis=1
)
inrix_full_gdf = tag(inrix_full_gdf, "inrix", ["tmc", "geometry"])

inrix_full_gdf = inrix_full_gdf.assign(inrix_seg_len_m=lambda df: df.geometry.length)

unique_tmc_ids = tmc_identification_df.tmc.unique().tolist()
inrix_filtered_by_tmc = inrix_full_gdf[inrix_full_gdf["tmc"].isin(unique_tmc_ids)]

# Name of the portal geometry column - retained to add back after sjoin_nearest()
PORTAL_GEOM_COL = portal_gdf.geometry.name


portal_w_inrix = (
    portal_gdf.sjoin_nearest(
        inrix_filtered_by_tmc,
        how="right",
        lsuffix="portal",
        rsuffix="inrix",
        distance_col="gpd_distance",
        max_distance=nearest_distance_limit,
    )[
        [
            "stationid",
            "tmc",
            "inrix_standardized_direction",
            "gpd_distance",
            "portal_direction",
        ]
    ]
    .dropna(
        subset=[
            "stationid",
            "tmc",
            "inrix_standardized_direction",
            "portal_direction",
            "gpd_distance",
        ]
    )
    .assign(
        # normalize types
        stationid=lambda df: df["stationid"].astype("Int64").astype(str),
        tmc=lambda df: df["tmc"].astype(str).str.strip(),
    )
    .merge(
        inrix_filtered_by_tmc[["tmc", "geometry"]].rename(
            columns={"geometry": "tmc_geom"}
        ),
        on="tmc",
        how="left",
    )
)

portal_w_inrix_same_dir = portal_w_inrix.query(
    "inrix_standardized_direction == portal_direction"
).copy()

assert (
    portal_w_inrix_same_dir["inrix_standardized_direction"]
    == portal_w_inrix_same_dir["portal_direction"]
).all()

portal_geom_df = (
    portal_gdf[["stationid", PORTAL_GEOM_COL]]
    .assign(stationid=lambda df: df["stationid"].astype("Int64").astype(str))
    .rename(columns={PORTAL_GEOM_COL: "portal_geom"})
)

# Merge back the portal geometries as its own column.
cand = portal_w_inrix_same_dir.merge(portal_geom_df, on="stationid", how="left")

# Make it a GeoDataFrame so vectorized geo ops work
cand_gdf = gpd.GeoDataFrame(cand, geometry="portal_geom", crs=portal_gdf.crs)

# Save Tmc geometries as a GeoSeries for easy access
tmc_series = gpd.GeoSeries(cand_gdf["tmc_geom"], index=cand_gdf.index, crs=cand_gdf.crs)

buf_m = 8
tmc_buf = tmc_series.buffer(buf_m)

cand_gdf["overlap_len"] = cand_gdf.geometry.intersection(
    tmc_buf, align=False
).length  # align = False to prevent surprise reindexing.

# %%
hungarian_result = hungarian_dist(cand_gdf, threshold=hung_threshold)
final_with_meta = (
    hungarian_result.merge(inrix_filtered_by_tmc[inrix_meta_cols], on="tmc")
    .merge(portal_gdf[portal_meta_cols], on="stationid")
    .drop_duplicates()[portal_meta_cols + inrix_meta_cols]
)

hungarian_result[["stationid", "tmc"]].to_csv(
    f"data/{year_str}_portal_inrix_spatial_join_hungarian_ids_only.csv", index=False
)

final_with_meta.to_csv(
    f"data/{year_str}_portal_inrix_spatial_join_hungarian.csv", index=False
)

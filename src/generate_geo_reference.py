"""
generate_geo_reference.py
--------------------------
Stage-2 optimisation (the ILP relief-allocation model) needs two pieces of
geographic reference data that are NOT present in the modelling dataset:

1. data/circle_centroids.csv
   Approximate latitude/longitude for every Revenue Circle (`object_id`).

2. data/relief_bases.csv
   Location and resource capacity (boats / food units / medical teams) of
   each relief base (e.g. SDRF/NDRF staging points, district HQs).

The project's modelling dataset (baseline_modeling_dataset_with_metadata.csv)
does not ship with GIS coordinates, so this script deterministically
generates realistic placeholder coordinates within Assam's bounding box and
a set of relief bases at well-known Assam towns.

*** This is clearly-labelled placeholder geography. ***
To use real data, replace the two output CSVs with actual Revenue Circle
centroids (e.g. exported from the GeoPandas shapefiles used in Stage 1 EDA)
and real base locations/capacities, keeping the same column names. No other
file needs to change — every downstream script (optimize_relief.py,
dashboard/app.py) only reads these two CSVs by column name.

Run once:
    python src/generate_geo_reference.py
"""

import hashlib

import pandas as pd

import config

# Assam's approximate bounding box
LAT_MIN, LAT_MAX = 24.2, 27.8
LON_MIN, LON_MAX = 89.8, 96.0

# Well-known Assam towns used as relief-base locations, with an assumed
# resource capacity for each. Replace with real SDRF/NDRF/ASDMA staging
# capacities when available.
RELIEF_BASES = [
    {"base_id": "BASE_GUWAHATI", "base_name": "Guwahati",   "latitude": 26.1445, "longitude": 91.7362, "boat_capacity": 600, "food_capacity": 12000, "medical_teams": 60},
    {"base_id": "BASE_DIBRUGARH", "base_name": "Dibrugarh", "latitude": 27.4728, "longitude": 94.9120, "boat_capacity": 400, "food_capacity": 8000, "medical_teams": 40},
    {"base_id": "BASE_JORHAT", "base_name": "Jorhat",       "latitude": 26.7509, "longitude": 94.2037, "boat_capacity": 350, "food_capacity": 7000, "medical_teams": 35},
    {"base_id": "BASE_SILCHAR", "base_name": "Silchar",     "latitude": 24.8333, "longitude": 92.7789, "boat_capacity": 400, "food_capacity": 8000, "medical_teams": 40},
    {"base_id": "BASE_TEZPUR", "base_name": "Tezpur",       "latitude": 26.6338, "longitude": 92.8000, "boat_capacity": 350, "food_capacity": 7000, "medical_teams": 35},
    {"base_id": "BASE_NAGAON", "base_name": "Nagaon",       "latitude": 26.3465, "longitude": 92.6840, "boat_capacity": 350, "food_capacity": 7000, "medical_teams": 35},
    {"base_id": "BASE_DHUBRI", "base_name": "Dhubri",       "latitude": 26.0210, "longitude": 89.9850, "boat_capacity": 300, "food_capacity": 6000, "medical_teams": 30},
    {"base_id": "BASE_BONGAIGAON", "base_name": "Bongaigaon", "latitude": 26.4770, "longitude": 90.5590, "boat_capacity": 300, "food_capacity": 6000, "medical_teams": 30},
    {"base_id": "BASE_LAKHIMPUR", "base_name": "Lakhimpur", "latitude": 27.2336, "longitude": 94.1050, "boat_capacity": 300, "food_capacity": 6000, "medical_teams": 30},
]
# NOTE: capacities represent total surge capacity available across the relief
# window (multiple boat trips / food-supply cycles / team-days), not a single
# one-off trip — consistent with the demand heuristic in config.py, which
# also estimates total affected population rather than a single instant.


def _deterministic_unit_floats(key: str, n: int = 2):
    """Turn a string key into n deterministic floats in [0, 1)."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    chunk = len(digest) // n
    return [int(digest[i * chunk:(i + 1) * chunk], 16) / 16 ** chunk for i in range(n)]


def build_circle_centroids() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DATASET_PATH, usecols=[config.ID_COLUMN])
    circle_ids = sorted(df[config.ID_COLUMN].unique())

    rows = []
    for circle_id in circle_ids:
        u_lat, u_lon = _deterministic_unit_floats(circle_id)
        lat = LAT_MIN + u_lat * (LAT_MAX - LAT_MIN)
        lon = LON_MIN + u_lon * (LON_MAX - LON_MIN)
        rows.append({config.ID_COLUMN: circle_id, "latitude": round(lat, 5), "longitude": round(lon, 5)})

    return pd.DataFrame(rows)


def build_relief_bases() -> pd.DataFrame:
    return pd.DataFrame(RELIEF_BASES)


def main():
    centroids = build_circle_centroids()
    centroids.to_csv(config.CENTROIDS_PATH, index=False)
    print(f"Wrote {len(centroids)} circle centroids to {config.CENTROIDS_PATH}")

    bases = build_relief_bases()
    bases.to_csv(config.BASES_PATH, index=False)
    print(f"Wrote {len(bases)} relief bases to {config.BASES_PATH}")


if __name__ == "__main__":
    main()

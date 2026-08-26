"""
optimize_relief.py
-------------------
Stage 2 — Prescriptive Engine.

Takes the Stage-1 output (outputs/predictions_latest.csv: predicted
probability + relief demand per Revenue Circle) and the geographic
reference data (data/circle_centroids.csv, data/relief_bases.csv) and
solves a Mixed-Integer Linear Program that decides how many boats, food
units and medical teams to send from each relief base to each
in-need circle.

Objective
    Minimise total *priority-weighted* unmet demand across all resource
    types, i.e. prefer leaving low-priority circles under-served over
    high-priority (high predicted-probability) circles under-served.

Constraints
    - Each base cannot allocate more of a resource than its capacity.
    - A base can only serve a circle within config.MAX_TRAVEL_KM
      (approximate straight-line distance).
    - Allocated resources for a circle cannot exceed that circle's need
      (no wasted over-allocation).

Run (after predict_impact.py and generate_geo_reference.py):
    python src/optimize_relief.py
"""

import math

import pandas as pd
import pulp

import config

RESOURCES = {
    "boats": {"demand_col": "boats_needed", "capacity_col": "boat_capacity"},
    "food_units": {"demand_col": "food_units_needed", "capacity_col": "food_capacity"},
    "medical_teams": {"demand_col": "medical_teams_needed", "capacity_col": "medical_teams"},
}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_inputs():
    predictions = pd.read_csv(config.PREDICTIONS_PATH)
    centroids = pd.read_csv(config.CENTROIDS_PATH)
    bases = pd.read_csv(config.BASES_PATH)

    circles = predictions[predictions["predicted_probability"] >= config.DEMAND_PROBABILITY_THRESHOLD].copy()
    circles = circles.merge(centroids, on=config.ID_COLUMN, how="left")
    return circles, bases


def build_distance_matrix(circles: pd.DataFrame, bases: pd.DataFrame) -> dict:
    distances = {}
    for _, circle in circles.iterrows():
        for _, base in bases.iterrows():
            d = haversine_km(circle["latitude"], circle["longitude"], base["latitude"], base["longitude"])
            distances[(circle[config.ID_COLUMN], base["base_id"])] = d
    return distances


def solve(circles: pd.DataFrame, bases: pd.DataFrame, distances: dict):
    circle_ids = circles[config.ID_COLUMN].tolist()
    base_ids = bases["base_id"].tolist()

    feasible_pairs = [
        (c, b) for c in circle_ids for b in base_ids if distances[(c, b)] <= config.MAX_TRAVEL_KM
    ]

    priority = circles.set_index(config.ID_COLUMN)["predicted_probability"].to_dict()

    prob = pulp.LpProblem("Assam_Flood_Relief_Allocation", pulp.LpMinimize)

    # Decision variables: units of each resource sent from base b to circle c.
    alloc = {
        (resource, c, b): pulp.LpVariable(f"alloc_{resource}_{c}_{b}", lowBound=0, cat="Integer")
        for resource in RESOURCES
        for (c, b) in feasible_pairs
    }

    # Unmet-demand variables (per circle, per resource) — what the objective minimises.
    unmet = {
        (resource, c): pulp.LpVariable(f"unmet_{resource}_{c}", lowBound=0, cat="Continuous")
        for resource in RESOURCES
        for c in circle_ids
    }

    # Objective: minimise priority-weighted unmet demand summed over all resources.
    prob += pulp.lpSum(
        (1.0 + priority[c]) * unmet[(resource, c)]
        for resource in RESOURCES
        for c in circle_ids
    )

    demand = circles.set_index(config.ID_COLUMN)

    # Demand-satisfaction constraints: allocated + unmet = required, per circle/resource.
    for resource, cols in RESOURCES.items():
        for c in circle_ids:
            required = float(demand.loc[c, cols["demand_col"]])
            served = pulp.lpSum(
                alloc[(resource, c, b)] for b in base_ids if (c, b) in feasible_pairs
            )
            prob += served + unmet[(resource, c)] == required, f"demand_{resource}_{c}"

    # Capacity constraints: a base cannot send more of a resource than it has.
    for resource, cols in RESOURCES.items():
        for b in base_ids:
            capacity = float(bases.set_index("base_id").loc[b, cols["capacity_col"]])
            sent = pulp.lpSum(
                alloc[(resource, c, b)] for c in circle_ids if (c, b) in feasible_pairs
            )
            prob += sent <= capacity, f"capacity_{resource}_{b}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]
    return prob, alloc, unmet, status, feasible_pairs


def collect_results(circles, bases, distances, alloc, unmet, feasible_pairs):
    rows = []
    for resource in RESOURCES:
        for (c, b) in feasible_pairs:
            qty = alloc[(resource, c, b)].value()
            if qty and qty > 1e-6:
                rows.append(
                    {
                        "resource": resource,
                        config.ID_COLUMN: c,
                        "base_id": b,
                        "units_allocated": round(qty),
                        "distance_km": round(distances[(c, b)], 1),
                    }
                )
    allocation_df = pd.DataFrame(rows)

    reachable = {c for (c, b) in feasible_pairs}

    summary_rows = []
    for resource, cols in RESOURCES.items():
        for _, circle in circles.iterrows():
            c = circle[config.ID_COLUMN]
            required = float(circle[cols["demand_col"]])
            unmet_qty = unmet[(resource, c)].value() or 0.0
            summary_rows.append(
                {
                    config.ID_COLUMN: c,
                    "impact_category": circle["impact_category"],
                    "predicted_probability": circle["predicted_probability"],
                    "priority_rank": circle["priority_rank"],
                    "resource": resource,
                    "required": required,
                    "allocated": round(required - unmet_qty, 2),
                    "unmet": round(unmet_qty, 2),
                    "fulfilment_pct": round(100 * (required - unmet_qty) / required, 1) if required > 0 else 100.0,
                    "within_base_range": c in reachable,
                }
            )
    summary_df = pd.DataFrame(summary_rows).sort_values(["priority_rank", "resource"])

    return allocation_df, summary_df


def main():
    circles, bases = load_inputs()
    if circles.empty:
        print("No circles meet the demand threshold — nothing to optimise.")
        return

    print(f"Optimising relief allocation for {len(circles)} in-need circles across {len(bases)} bases...")
    distances = build_distance_matrix(circles, bases)

    prob, alloc, unmet, status, feasible_pairs = solve(circles, bases, distances)
    print(f"Solver status: {status}")

    allocation_df, summary_df = collect_results(circles, bases, distances, alloc, unmet, feasible_pairs)

    allocation_df.to_csv(config.ALLOCATION_PATH, index=False)
    summary_df.to_csv(config.ALLOCATION_SUMMARY_PATH, index=False)

    print(f"\nSaved detailed base->circle allocation to {config.ALLOCATION_PATH}")
    print(f"Saved per-circle fulfilment summary to {config.ALLOCATION_SUMMARY_PATH}")

    overall_fulfilment = (
        100 * (summary_df["required"].sum() - summary_df["unmet"].sum()) / summary_df["required"].sum()
        if summary_df["required"].sum() > 0
        else 100.0
    )
    print(f"\nOverall demand fulfilled: {overall_fulfilment:.1f}%")

    unreachable = sorted(summary_df.loc[~summary_df["within_base_range"], config.ID_COLUMN].unique())
    if unreachable:
        print(
            f"\n{len(unreachable)} circle(s) are beyond {config.MAX_TRAVEL_KM} km of every relief base "
            f"and cannot be served without a new/closer staging point: {unreachable}"
        )

    print("\nLowest-fulfilment circles (bottleneck areas needing more capacity):")
    worst = summary_df.sort_values("fulfilment_pct").head(5)
    print(worst[[config.ID_COLUMN, "resource", "required", "allocated", "fulfilment_pct", "within_base_range"]].to_string(index=False))


if __name__ == "__main__":
    main()

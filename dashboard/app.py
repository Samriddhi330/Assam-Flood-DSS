"""
dashboard/app.py
-----------------
Streamlit dashboard for the Assam Flood DSS. Visualises:

  - Predicted flood-impact probability / category per Revenue Circle
  - Why each circle was flagged (SHAP top reasons)
  - The relief base -> circle allocation plan and unmet demand
  - An interactive map (Folium) of circles, bases and allocation lines

This reads the CSVs already produced by src/pipeline_run.py — it does not
retrain or re-optimise anything itself, so it opens instantly. If the
outputs don't exist yet, it tells the user to run the pipeline first.

Run:
    python src/pipeline_run.py      # once, to generate the CSVs
    streamlit run dashboard/app.py
"""

import os
import sys

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config  # noqa: E402

st.set_page_config(page_title="Assam Flood DSS", layout="wide")


@st.cache_data
def load_outputs():
    missing = [
        p
        for p in [
            config.PREDICTIONS_PATH,
            config.ALLOCATION_SUMMARY_PATH,
            config.ALLOCATION_PATH,
            config.CENTROIDS_PATH,
            config.BASES_PATH,
        ]
        if not os.path.exists(p)
    ]
    if missing:
        return None

    predictions = pd.read_csv(config.PREDICTIONS_PATH)
    summary = pd.read_csv(config.ALLOCATION_SUMMARY_PATH)
    allocation = pd.read_csv(config.ALLOCATION_PATH)
    centroids = pd.read_csv(config.CENTROIDS_PATH)
    bases = pd.read_csv(config.BASES_PATH)
    return predictions, summary, allocation, centroids, bases


def category_color(category: str) -> str:
    return {"High": "red", "Medium": "orange", "Low": "green"}.get(category, "gray")


def build_map(predictions, allocation, centroids, bases) -> folium.Map:
    merged = predictions.merge(centroids, on=config.ID_COLUMN, how="left")

    center_lat = merged["latitude"].mean()
    center_lon = merged["longitude"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")

    # Relief bases
    for _, base in bases.iterrows():
        folium.Marker(
            location=[base["latitude"], base["longitude"]],
            tooltip=f"Relief Base: {base['base_name']}",
            popup=(
                f"<b>{base['base_name']}</b><br>"
                f"Boats: {base['boat_capacity']}<br>"
                f"Food units: {base['food_capacity']}<br>"
                f"Medical teams: {base['medical_teams']}"
            ),
            icon=folium.Icon(color="blue", icon="home"),
        ).add_to(fmap)

    # Revenue circles, coloured by predicted impact category
    for _, row in merged.iterrows():
        if pd.isna(row["latitude"]):
            continue
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5 + 6 * row["predicted_probability"],
            color=category_color(row["impact_category"]),
            fill=True,
            fill_opacity=0.75,
            tooltip=(
                f"{row[config.ID_COLUMN]} — {row['impact_category']} "
                f"({row['predicted_probability']:.2f})"
            ),
            popup=row.get("top_reasons", ""),
        ).add_to(fmap)

    # Allocation lines: base -> circle for the largest resource flow only,
    # to keep the map readable.
    if not allocation.empty:
        boats_alloc = allocation[allocation["resource"] == "boats"]
        base_lookup = bases.set_index("base_id")[["latitude", "longitude"]]
        circle_lookup = centroids.set_index(config.ID_COLUMN)[["latitude", "longitude"]]
        for _, row in boats_alloc.iterrows():
            if row["base_id"] not in base_lookup.index or row[config.ID_COLUMN] not in circle_lookup.index:
                continue
            b = base_lookup.loc[row["base_id"]]
            c = circle_lookup.loc[row[config.ID_COLUMN]]
            folium.PolyLine(
                locations=[[b["latitude"], b["longitude"]], [c["latitude"], c["longitude"]]],
                color="blue",
                weight=1,
                opacity=0.4,
            ).add_to(fmap)

    return fmap


def main():
    st.title("Assam Flood — Explainable Impact Assessment & Relief Allocation")
    st.caption(
        "Stage 1 (ML + SHAP) predicts next-month flood impact per Revenue Circle. "
        "Stage 2 (Integer Linear Programming) allocates boats, food and medical "
        "teams from relief bases to the circles that need them most."
    )

    data = load_outputs()
    if data is None:
        st.warning(
            "No pipeline outputs found yet. Run `python src/pipeline_run.py` from the "
            "project root first, then reload this page."
        )
        return

    predictions, summary, allocation, centroids, bases = data

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue circles scored", len(predictions))
    col2.metric("High-impact circles", int((predictions["impact_category"] == "High").sum()))
    col3.metric("Medium-impact circles", int((predictions["impact_category"] == "Medium").sum()))
    required_total = summary["required"].sum()
    fulfilled_total = summary["allocated"].sum()
    col4.metric(
        "Overall demand fulfilled",
        f"{100 * fulfilled_total / required_total:.1f}%" if required_total > 0 else "N/A",
    )

    st.subheader("Flood impact & relief map")
    fmap = build_map(predictions, allocation, centroids, bases)
    st_folium(fmap, width=None, height=520)
    st.caption(
        "Circle size/colour = predicted flood-impact probability (green = low, "
        "orange = medium, red = high). Blue markers = relief bases. Thin blue "
        "lines = planned boat allocations."
    )

    st.subheader("Relief priority ranking")
    display_cols = [
        config.ID_COLUMN,
        "predicted_probability",
        "impact_category",
        "priority_rank",
        "predicted_affected_population",
        "boats_needed",
        "food_units_needed",
        "medical_teams_needed",
        "top_reasons",
    ]
    st.dataframe(
        predictions[display_cols].sort_values("priority_rank"),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Relief allocation — demand fulfilment by circle")
    pivot = summary.pivot_table(
        index=[config.ID_COLUMN, "impact_category", "priority_rank"],
        columns="resource",
        values="fulfilment_pct",
    ).reset_index().sort_values("priority_rank")
    st.dataframe(pivot, use_container_width=True, hide_index=True)

    st.subheader("Detailed base → circle allocation")
    st.dataframe(allocation.sort_values(["object_id", "resource"]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

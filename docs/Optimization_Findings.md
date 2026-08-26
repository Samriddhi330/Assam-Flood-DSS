# Optimization & Explainability Findings — Assam Flood DSS

This document continues directly from `docs/Modeling_Findings.md` and covers
items 5–10 of its "Next Steps" section: SHAP explainability, converting
predicted impact into relief demand, the Integer Linear Programming (ILP)
optimisation model, and the integrated two-stage pipeline.

---

## 1. Explainable AI — TreeSHAP (`src/explain_shap.py`)

The selected Stage-1 model (chosen by F1 Score on the temporal test split,
per `docs/Modeling_Findings.md` section 20) is wrapped in a
`shap.TreeExplainer`.

- **Global explanation**: mean absolute SHAP value per feature across all
  Revenue Circles in the most recent month, saved to
  `outputs/shap_global_importance.csv`.
- **Local explanation**: per-circle SHAP contributions and a human-readable
  "top reasons" string (e.g. `"max_rain contributed +25%; mean_ndbi
  contributed -11%; impact contributed -9%"`), saved to
  `outputs/shap_local_explanations.csv`.

This matches the project write-up's example output format: *"Circle X needs
priority not just due to rainfall, but because ... of its score is driven
by ..."*.

---

## 2. From Probability to Relief Demand (`src/predict_impact.py`)

The dataset does not contain a directly usable "relief demand" label — using
one (affected population, camp inmates, etc.) as a *feature* would leak the
outcome, which is why `docs/Modeling_Findings.md` section 18 explicitly
excludes those columns from the model's inputs. They can, however, be used
*after* prediction to translate a probability into an operational demand
estimate.

The heuristic used (see `src/config.py` for the exact constants):

```
predicted_affected_population = predicted_probability × sum_population
boats_needed          = affected_population / PEOPLE_PER_BOAT
food_units_needed     = affected_population / PEOPLE_PER_FOOD_PACKET_BATCH
medical_teams_needed  = affected_population / PEOPLE_PER_MEDICAL_TEAM
```

This is intentionally simple and clearly isolated in one function
(`demand_from_probability`) so it can be replaced with real ASDMA
relief-camp / affected-population figures without changing any other part
of the pipeline.

Circles are also assigned an `impact_category` (Low / Medium / High) by
thresholding the predicted probability, and a `priority_rank` (1 = most
urgent), which together reproduce the "Predicted Impact / Priority / Main
Reasons" table format from the original project write-up.

---

## 3. Geographic Reference Data (`src/generate_geo_reference.py`)

Stage 2 needs coordinates for Revenue Circles and relief bases, which the
modelling dataset does not include. `src/generate_geo_reference.py`
generates:

- `data/circle_centroids.csv` — one deterministic placeholder lat/lon per
  `object_id`, generated within Assam's bounding box.
- `data/relief_bases.csv` — nine relief bases at well-known Assam towns
  (Guwahati, Dibrugarh, Jorhat, Silchar, Tezpur, Nagaon, Dhubri,
  Bongaigaon, Lakhimpur) with assumed boat / food / medical-team capacities.

**These are clearly-labelled placeholder values for demonstration.** To move
to production, replace both CSVs with real Revenue Circle centroids (e.g.
exported from the GeoPandas shapefiles referenced in the EDA stage) and real
base locations/capacities from ASDMA/SDRF, keeping the same column names —
no other script needs to change.

---

## 4. Relief Optimisation — Integer Linear Program (`src/optimize_relief.py`)

**Sets**
- Circles `i` with predicted probability ≥ `DEMAND_PROBABILITY_THRESHOLD`
  (0.20 by default) — i.e. circles worth planning relief for.
- Relief bases `j`.
- Resources `r` ∈ {boats, food_units, medical_teams}.

**Decision variables**
- `x[r,i,j] ≥ 0` (integer): units of resource `r` sent from base `j` to
  circle `i`.
- `unmet[r,i] ≥ 0` (continuous): shortfall of resource `r` at circle `i`.

**Objective** — minimise total priority-weighted unmet demand:

```
minimize   Σ (1 + predicted_probability_i) × unmet[r,i]   over all r, i
```

Weighting unmet demand by `(1 + probability_i)` means the solver prefers to
leave a low-priority circle short before a high-priority one — this is what
operationalises "flood severity alone is not enough; direct help to where
it is needed most."

**Constraints**
1. Demand balance: `Σ_j x[r,i,j] + unmet[r,i] = required[r,i]` for every
   circle/resource.
2. Base capacity: `Σ_i x[r,i,j] ≤ capacity[r,j]` for every base/resource.
3. Reachability: `x[r,i,j] = 0` whenever the great-circle distance between
   circle `i` and base `j` exceeds `MAX_TRAVEL_KM` (200 km by default).

Solved with the open-source CBC solver via PuLP.

**Outputs**
- `outputs/relief_allocation.csv` — every non-zero `base → circle` shipment.
- `outputs/relief_allocation_summary.csv` — per-circle, per-resource
  required / allocated / unmet / fulfilment % / whether any base is within
  range, sorted by priority rank.

The run log also lists any circles that are beyond `MAX_TRAVEL_KM` of every
base — a direct, actionable signal that a new staging point is needed
there, independent of resource scarcity.

---

## 5. Sample Result

On the bundled dataset (180 Revenue Circles, most recent available month,
9 placeholder relief bases):

- 43 circles crossed the demand threshold.
- 4 circles had no base within 200 km (flagged as unreachable, not just
  under-resourced).
- Overall demand fulfilled across boats/food/medical teams: **~66%**,
  concentrated shortfalls in boats and medical teams — consistent with the
  real-world "boat and medical-team shortage" narrative described in the
  project background.

Exact numbers will shift slightly on re-run if the dataset is updated
(a new "latest month" becomes available) or if the placeholder geography is
replaced with real coordinates/capacities.

---

## 6. Full Pipeline (`src/pipeline_run.py`)

Runs steps 0–4 above in order and prints a summary. This is the single
command that reproduces everything in `outputs/` from the raw dataset in
`data/`:

```bash
pip install -r requirements.txt
python src/pipeline_run.py
streamlit run dashboard/app.py
```

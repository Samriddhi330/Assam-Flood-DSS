# EDA Findings — Assam Flood DSS

## 1. Project Context

This notebook contains the Exploratory Data Analysis (EDA), data-quality investigation, target construction, and prediction-feature selection performed for the Assam Flood DSS project.

The objective of the prediction component is to determine whether a Revenue Circle is likely to experience **flood impact in the following month**, using information that is available for the current prediction period.

The feature-selection process was guided by one primary principle:

> **A variable should only be used as a prediction feature if it could realistically be known at the time the prediction is made.**

This principle was used to identify potential target leakage, distinguish genuine predictors from flood consequences, and separate the baseline early-warning model from variables that may be useful in secondary DSS models.

---

# 2. Dataset Overview

* **Geographic unit:** Revenue Circle
* **Number of Revenue Circles:** 180
* **Time period:** April 2021 – May 2026
* **Total observations:** 11,160
* **Initial candidate prediction features:** 37
* **Observations available after constructing the one-month-ahead target:** 10,980

The dataset contains a combination of:

* rainfall and environmental variables
* satellite-derived environmental indicators
* population and household exposure
* infrastructure information
* agricultural exposure
* topographical and hydrological characteristics
* historical flood impact
* flood/inundation observations
* flood-response and damage-related information

---

# 3. Target Construction

The original dataset contains the variable:

```text
impact
```

which represents flood impact for the corresponding Revenue Circle and time period.

For prediction, a one-month-ahead target was constructed:

```text
next_month_impact
```

Thus, the modelling relationship is conceptually:

```text
Current-period information
          ↓
     Prediction
          ↓
Next-month flood impact
```

The target was constructed by temporally shifting `impact` within each Revenue Circle.

The final target distribution was:

| Target | Observations | Percentage |
| ------ | -----------: | ---------: |
| 0      |       10,103 |     92.01% |
| 1      |          877 |      7.99% |

There were initially **180 observations with missing target values**, corresponding to periods for which the following month's impact was unavailable. These rows were excluded from the final modelling dataset.

Therefore:

* **Final modelling observations:** 10,980
* **Positive observations:** 877
* **Negative observations:** 10,103
* **Positive-class proportion:** 7.99%

The target is therefore **highly imbalanced**, which should be considered during later model evaluation.

---

# 4. Candidate Feature Set

Initially, **37 variables** were identified as potential prediction features.

These variables covered several categories.

## 4.1 Hazard / Environmental Variables

* `max_rain`
* `mean_rain`
* `sum_rain`
* `mean_ndvi`
* `mean_ndbi`
* `riverlevel_mean`
* `riverlevel_min`
* `riverlevel_max`
* `inundation_pct`
* `inundation_intensity_mean`
* `inundation_intensity_mean_nonzero`
* `inundation_intensity_sum`
* `flooded_vegetation`
* `water`
* `trees`
* `crops`
* `built_area`
* `bare_ground`
* `rangeland`
* `clouds`

## 4.2 Population / Exposure / Vulnerability

* `sum_population`
* `sum_aged_population`
* `sum_young_population`
* `mean_sex_ratio`
* `total_hhd`
* `net_sown_area_in_hac`

## 4.3 Infrastructure

* `schools_count`
* `health_centres_count`
* `road_length`
* `rail_length`
* `rail_count`

## 4.4 Physical Geography

* `distance_from_river`
* `drainage_density`
* `elevation_mean`
* `slope_mean`
* `mean_cn`

## 4.5 Historical Impact

* `impact`

The original candidate set therefore contained variables representing both **potential predictors** and variables that are actually observations of the flood event or its consequences.

---

# 5. Initial Feature-Screening Philosophy

The 37 variables were initially classified according to:

1. Prediction-time availability
2. Leakage risk
3. Data coverage
4. Zero-value behaviour
5. Physical relevance
6. Relationship to flood impact
7. Whether the variable represents a cause/predictor or a consequence of flooding

The initial screening produced:

| Decision               | Count |
| ---------------------- | ----: |
| KEEP                   |    17 |
| CHECK TIMING           |    11 |
| SECONDARY              |     5 |
| CHECK ZERO MEANING     |     2 |
| REMOVE                 |     1 |
| KEEP AS LAGGED FEATURE |     1 |

This screening was subsequently refined through additional validation.

---

# 6. Rainfall Findings

The rainfall variables were retained as core predictors:

* `max_rain`
* `mean_rain`
* `sum_rain`

These variables represent direct rainfall conditions and are broadly available across the dataset.

They are among the most defensible hazard predictors because rainfall information can realistically be available before predicting the following month's flood impact.

The three variables represent different aspects of rainfall:

* `max_rain` — maximum rainfall intensity/measurement
* `mean_rain` — average rainfall
* `sum_rain` — cumulative rainfall

For future modelling, the temporal relationship should remain explicit:

```text
Current-period rainfall → next-month impact
```

rather than accidentally incorporating rainfall information from the target month.

---

# 7. Population and Exposure Findings

The following population/exposure variables were retained:

* `sum_population`
* `sum_aged_population`
* `sum_young_population`
* `mean_sex_ratio`
* `total_hhd`
* `net_sown_area_in_hac`

These variables describe the population and assets potentially exposed to flooding rather than reporting flood consequences.

The population variables had complete coverage in the analysed dataset.

However, several population variables are naturally related to one another. For example:

```text
sum_population
sum_aged_population
sum_young_population
```

may contain overlapping information.

They were therefore retained as candidate baseline features, while future modelling can determine whether all of them provide additional predictive value.

---

# 8. Infrastructure Findings

The following infrastructure variables were retained:

* `schools_count`
* `health_centres_count`
* `road_length`

These represent infrastructure exposure and vulnerability and are therefore conceptually appropriate for a flood-impact prediction model.

Two railway variables require additional interpretation:

* `rail_length`
* `rail_count`

Both had:

* 7,316 nonzero observations
* 65.56% nonzero coverage
* 3,844 zero observations
* 0 missing values

The important issue is that a zero could mean either:

```text
No railway exists in the Revenue Circle
```

or potentially:

```text
Railway information was unavailable / represented as zero
```

Source-level validation is therefore required before treating the zero values as genuine absence.

They were consequently **not included in the current baseline**.

---

# 9. Physical Geography Findings

The following physical/geographical variables were retained:

* `distance_from_river`
* `drainage_density`
* `elevation_mean`
* `slope_mean`
* `mean_cn`

These variables describe relatively stable physical characteristics of the Revenue Circle.

They are useful from a flood-susceptibility perspective because they represent characteristics of the environment rather than consequences of a flood event.

Their usefulness should therefore not be judged only by simple correlation with the target. They may provide predictive information through interactions and nonlinear relationships in later models.

---

# 10. NDVI and NDBI

The following variables were ultimately included in the baseline:

* `mean_ndvi`
* `mean_ndbi`

Both variables had complete coverage:

* 11,160 observations
* 100% coverage
* 0 missing values
* no zero values

Their observed ranges were:

### NDVI

* Minimum: approximately -0.099
* Maximum: approximately 0.888
* Mean: approximately 0.472

### NDBI

* Minimum: approximately -0.580
* Maximum: approximately 0.129
* Mean: approximately -0.123

Both were considered potentially useful environmental predictors.

However, their **source/acquisition timing is important**. Satellite-derived information must correspond to information that would genuinely be available before the target prediction.

Source-level investigation identified them as sufficiently defensible to retain in the baseline, while their temporal interpretation should still be documented and verified where possible.

---

# 11. Land-Cover Variables

The following variables were investigated:

* `water`
* `trees`
* `crops`
* `built_area`
* `bare_ground`
* `rangeland`

Their coverage was substantially lower than the rainfall, population, geography, and NDVI/NDBI variables.

Observed nonzero coverage was:

| Variable      | Nonzero observations | Nonzero % |
| ------------- | -------------------: | --------: |
| `bare_ground` |                4,947 |    44.33% |
| `rangeland`   |                5,673 |    50.83% |
| `water`       |                5,907 |    52.93% |
| `trees`       |                5,940 |    53.23% |
| `crops`       |                5,940 |    53.23% |
| `built_area`  |                5,940 |    53.23% |

There were no missing values, but many observations were exactly zero.

The zero values cannot automatically be interpreted as missing data because they may represent genuine absence of a land-cover category.

However, source documentation did not provide sufficient confidence regarding the exact acquisition/timing relationship with the prediction period.

Therefore these variables were classified as:

> **TIMING UNCONFIRMED / SOURCE VALIDATION REQUIRED**

They are not part of the current baseline feature matrix.

---

# 12. River-Level Variables

The following variables were investigated:

* `riverlevel_mean`
* `riverlevel_min`
* `riverlevel_max`

Their coverage was substantially lower:

| Variable          | Nonzero observations | Nonzero % |
| ----------------- | -------------------: | --------: |
| `riverlevel_min`  |                1,881 |    16.85% |
| `riverlevel_mean` |                1,898 |    17.01% |
| `riverlevel_max`  |                1,898 |    17.01% |

All three had zero missing values, but approximately **83% of observations were zero**.

There was also a serious data-quality concern with the observed maximum:

```text
riverlevel_max = 75,030
```

compared with:

```text
riverlevel_mean ≈ 11.23
riverlevel_min ≈ 9.61
```

This requires careful interpretation and potential source validation.

More importantly, the analysis could not establish with sufficient confidence that these river-level observations were consistently available at the required prediction cutoff.

Therefore:

```text
riverlevel_mean
riverlevel_min
riverlevel_max
```

were **excluded from the strict baseline model**.

They may be reconsidered in a future operational model if their temporal availability and measurement semantics are established.

---

# 13. Inundation Variables

The following variables were excluded from the strict early-warning baseline:

* `inundation_pct`
* `inundation_intensity_mean`
* `inundation_intensity_mean_nonzero`
* `inundation_intensity_sum`

These variables describe the current flood/inundation state.

Their use could therefore create a conceptual mismatch with a genuine early-warning model:

```text
Flood is already occurring
        ↓
Current inundation is measured
        ↓
Use inundation to predict future impact
```

Such variables may still be useful for a different modelling question:

> **Given that flooding is already occurring, will flood impact persist into the following period?**

Therefore they have been classified as **secondary-model features**, rather than discarded completely.

---

# 14. Flooded Vegetation

`flooded_vegetation` was also excluded from the baseline.

The variable is closely associated with an already-observed flood condition rather than a stable pre-event characteristic.

It is therefore more appropriate for an **active-flood persistence model** than for the strict early-warning baseline.

It has consequently been classified as:

> **SECONDARY MODEL**

---

# 15. Clouds

`clouds` was removed from the feature set.

The variable was extremely sparse:

* Nonzero observations: 93
* Total observations: 11,160

This corresponds to less than 1% nonzero observations.

It was therefore not considered a meaningful core predictor for the baseline flood-impact model.

---

# 16. Historical Impact as a Lagged Feature

The original `impact` variable was retained, but it must be interpreted correctly.

For the one-month-ahead prediction setup, it represents **historical/current-period impact used to predict subsequent impact**.

Conceptually:

```text
Current-period impact
          ↓
    next-month impact
```

Therefore it should be treated as a **lagged historical feature**, not as the target itself.

This distinction is critical.

The model must never use the target month's impact as a predictor.

The intended relationship is:

```text
Previous/current-period impact → next-month impact
```

not:

```text
Next-month impact → next-month impact
```

---

# 17. Target Leakage Considerations

Several variables from the original dataset were considered unsuitable as direct predictors because they represent consequences of flooding.

Examples include variables related to:

* population affected
* animals affected
* animals washed away
* crop damage
* relief camps
* relief centres
* camp inmates
* houses damaged
* human lives lost
* expenditure
* government relief
* tenders
* repair/restoration
* immediate measures

These variables may be valuable for the broader DSS, but they do not belong in the strict flood-impact prediction baseline if their values become known only after or during the flood response.

In particular, `Population_affected_Total` should not be used as a predictor because the target itself is derived from flood impact.

This would introduce direct target leakage.

The general rule adopted was:

> **Flood consequences and response variables should not be treated as pre-event prediction features.**

They may instead be useful in separate DSS components such as:

* damage assessment
* resource allocation
* response planning
* preparedness analysis

---

# 18. Administrative / Identifier Variables

The dataset contains several geographic/entity identifiers:

* `object_id`
* `district`
* `rc_area`
* `revenue_ci`
* `dtname`
* `dtcode11`

These were not included directly in the baseline feature matrix.

In particular:

* `object_id` is an identifier and should not be treated as a numerical predictor.
* Administrative codes such as `dtcode11` should not be blindly passed into the model.
* Raw geographic identifiers can allow a model to memorize locations rather than learn generalizable physical relationships.

These variables remain useful for grouping, temporal alignment, validation, and interpretation.

---

# 19. Final Feature-Selection Decision

After the initial screening and subsequent validation, the final baseline feature set contains **20 features**.

## Core hazard features

```text
max_rain
mean_rain
sum_rain
```

## Environmental features

```text
mean_ndvi
mean_ndbi
```

## Population / exposure

```text
sum_population
sum_aged_population
sum_young_population
mean_sex_ratio
total_hhd
net_sown_area_in_hac
```

## Infrastructure

```text
schools_count
health_centres_count
road_length
```

## Physical geography

```text
distance_from_river
drainage_density
elevation_mean
slope_mean
mean_cn
```

## Historical feature

```text
impact
```

Therefore:

> **Final baseline feature count = 20**

---

# 20. Features Requiring Further Validation

The following variables were not included in the current baseline because their interpretation or timing requires additional source-level validation:

```text
water
trees
crops
built_area
bare_ground
rangeland
rail_length
rail_count
```

The earlier source-level validation identified:

* `mean_ndvi` → retained
* `mean_ndbi` → retained
* land-cover variables → timing remains insufficiently confirmed
* railway variables → meaning of zero remains insufficiently confirmed

These variables can be reconsidered in future model iterations once their source metadata and temporal semantics are established.

---

# 21. Secondary-Model Features

The following variables were intentionally separated from the strict baseline:

```text
inundation_pct
inundation_intensity_mean
inundation_intensity_mean_nonzero
inundation_intensity_sum
flooded_vegetation
```

These should not necessarily be considered "bad" features.

Instead, they answer a different modelling question.

A potential secondary model could investigate:

> **Flood persistence / active-flood forecasting**

where current flood-state information is intentionally available.

This separation allows the DSS to distinguish between:

### Early-warning model

```text
Can we predict next-month impact using information
available before/during the prediction cutoff?
```

and

### Flood-persistence model

```text
Given that flooding is currently occurring,
is significant impact likely to continue?
```

---

# 22. Final Modelling Dataset

After constructing the one-month-ahead target and removing rows for which the target could not be determined:

```text
Feature matrix shape: (10,980, 20)
Target shape:         (10,980,)
```

The final target distribution is:

```text
0 → 10,103
1 →    877
```

with:

```text
Positive class = 7.99%
```

The baseline feature matrix contains:

* **20 features**
* **10,980 observations**
* **0 missing values**
* all features numeric

The feature data types were validated and no non-numeric baseline features remained.

---

# 23. Duplicate Feature-Row Investigation

A duplicate-row investigation was performed on the baseline feature data.

The analysis identified:

```text
Duplicate feature rows: 229
```

However, importantly:

```text
Unique object_id + timeperiod combinations among duplicates: 0
```

This means the repeated feature rows are **not duplicate observations of the same Revenue Circle and time period**.

Several repeated rows correspond to different time periods for the same Revenue Circle and are therefore legitimate repeated observations in a panel/time-series dataset.

For example, the same Revenue Circle may have stable geographic, demographic, or infrastructure characteristics across multiple months.

Therefore:

> **The 229 duplicate feature rows should not automatically be deleted.**

They represent repeated feature configurations across different observations rather than confirmed duplicate records.

This investigation should be preserved as a data-quality finding for the team.

---

# 24. Final Data-Quality Summary

The final baseline dataset was checked for:

* missing values
* non-numeric features
* infinite values
* duplicate feature rows
* target availability
* target imbalance
* feature coverage
* zero-value behaviour
* temporal availability
* leakage risk
* source-level feature validity

The final baseline feature matrix has:

```text
Shape:              (10,980, 20)
Missing X values:   0
Missing y values:   0
Infinite values:    0
Non-numeric X:      None
```

The dataset does contain repeated feature configurations, but these were investigated and were not found to correspond to duplicate `object_id + timeperiod` observations.

---

# 25. Key EDA Conclusions

The main conclusions from the EDA and feature-selection analysis are:

### 1. The prediction problem is highly imbalanced

Only **7.99%** of the final observations correspond to positive next-month flood impact.

Therefore, future model evaluation should not rely on accuracy alone.

### 2. Rainfall is a strong and defensible baseline predictor group

`max_rain`, `mean_rain`, and `sum_rain` are broadly available and directly represent hydrometeorological conditions.

### 3. Stable geography and exposure variables are appropriate baseline predictors

Population, infrastructure, topography, drainage, river proximity, and runoff-related characteristics provide information about the vulnerability and susceptibility of Revenue Circles.

### 4. Historical impact is useful but must remain temporally aligned

`impact` is retained as a lagged feature because previous flood impact can provide information about subsequent impact.

Its temporal alignment must be preserved to avoid leakage.

### 5. Current flood-state variables should not automatically enter the early-warning model

Inundation and flooded-vegetation variables describe flooding that is already occurring.

They are therefore separated into a potential secondary flood-persistence model.

### 6. Response variables belong to the DSS, but not necessarily the prediction model

Relief, expenditure, damage, and tender-related variables can be valuable for downstream decision-support functions, but using them directly in the early-warning model could introduce leakage and mix prediction with post-event response.

### 7. Source timing matters

Satellite-derived and land-cover variables cannot be judged only by their statistical values.

Their acquisition period must be compatible with the prediction cutoff.

### 8. Zero values require semantic interpretation

For variables such as railway infrastructure and land-cover categories, zero does not automatically mean missing data.

The source definition must determine whether zero represents genuine absence.

### 9. The final baseline is intentionally conservative

The baseline prioritizes features that are reasonably defensible from a prediction-time perspective rather than simply maximizing the number of available variables.

---

# 26. Final Feature-Selection Summary

| Feature group                                    | Decision                                                         |
| ------------------------------------------------ | ---------------------------------------------------------------- |
| Rainfall                                         | Included in baseline                                             |
| NDVI / NDBI                                      | Included in baseline                                             |
| Population / exposure                            | Included in baseline                                             |
| Core infrastructure                              | Included in baseline                                             |
| Physical geography                               | Included in baseline                                             |
| Historical `impact`                              | Included as lagged feature                                       |
| Land-cover variables                             | Awaiting source/timing validation                                |
| Railway variables                                | Awaiting zero-value interpretation                               |
| River-level variables                            | Excluded from current baseline pending timing/data validation    |
| Inundation variables                             | Reserved for secondary model                                     |
| Flooded vegetation                               | Reserved for secondary model                                     |
| Clouds                                           | Excluded due to extreme sparsity                                 |
| Damage / relief / expenditure / tender variables | Not used as baseline predictors due to leakage/response concerns |
| Administrative identifiers                       | Not used directly as model features                              |

---

# 27. Handoff to the Modelling Team

EDA and prediction-feature preparation are complete.

The modelling stage should use the **20-feature baseline dataset** established above.

The modelling team should be aware of the following:

1. The target is `next_month_impact`.
2. The positive class represents only 7.99% of observations.
3. `impact` is intended as a historical/lagged predictor.
4. The 20 baseline features contain no missing values.
5. The baseline contains only numeric features.
6. Duplicate feature configurations were investigated and should not automatically be removed.
7. River-level variables are currently excluded.
8. Land-cover and railway variables require further source validation before inclusion.
9. Inundation/flooded-vegetation variables are reserved for a separate model.
10. Response/damage variables should not be introduced into the baseline without carefully establishing prediction-time availability.

The next stage is **model training and evaluation**, including appropriate handling of class imbalance and time-aware validation.

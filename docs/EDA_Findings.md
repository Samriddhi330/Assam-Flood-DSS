# EDA Findings — Assam Flood DSS

## 1. Dataset Overview

- Revenue Circles: 180
- Time period: April 2021 – May 2026
- Observations used for one-month-ahead prediction: 10,980

## 2. Target

The prediction target is next-month flood impact.

Next-month impact distribution:

- 0: 10,103
- 1: 877

Positive class proportion: 7.99%

## 3. Candidate Prediction Features

37 candidate features were identified.

### Environmental
- max_rain
- mean_rain
- sum_rain
- mean_ndvi
- mean_ndbi
- ...

### Exposure / Vulnerability
- sum_population
- sum_aged_population
- ...
  
### Infrastructure
- schools_count
- health_centres_count
- road_length
- ...

### Geographic
- distance_from_river
- drainage_density
- elevation_mean
- slope_mean
- mean_cn

## 4. Data Quality

All 37 candidate prediction features have 100% coverage
in the final prediction dataset.

## 5. Important EDA Findings

[put your actual findings here]

## 6. Prediction Dataset

The final one-month-ahead dataset contains:

- 10,980 observations
- 180 Revenue Circles
- April 2021 – April 2026 prediction periods

## 7. Model Training Handoff

EDA and prediction-feature preparation are complete.

The next stage is model training and evaluation.

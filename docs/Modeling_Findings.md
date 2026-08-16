# Modeling Findings — Assam Flood DSS

## 1. Modeling Objective

The objective of the modeling stage is to develop a machine learning model that predicts whether a Revenue Circle is likely to experience flood impact in the following month.

The modeling problem is therefore formulated as:

    Current-period information
              ↓
       Machine Learning Model
              ↓
    Next-month flood impact

The target variable is:

    `next_month_impact`

The final baseline feature set contains 20 prediction features selected during the EDA and feature-screening stage.

---

# 2. Final Modeling Dataset

The modeling dataset is based on the one-month-ahead prediction dataset created during EDA.

The baseline feature matrix contains:

- 20 prediction features
- 1 target variable: `next_month_impact`

Additional identification/time variables such as:

- `object_id`
- `timeperiod`

are retained separately for temporal ordering, grouping, validation and interpretation, but are not provided directly to the machine learning model.

The 20 prediction features are:

### Hazard Features

- `max_rain`
- `mean_rain`
- `sum_rain`

### Environmental Features

- `mean_ndvi`
- `mean_ndbi`

### Population / Exposure Features

- `sum_population`
- `sum_aged_population`
- `sum_young_population`
- `mean_sex_ratio`
- `total_hhd`
- `net_sown_area_in_hac`

### Infrastructure Features

- `schools_count`
- `health_centres_count`
- `road_length`

### Physical Geography Features

- `distance_from_river`
- `drainage_density`
- `elevation_mean`
- `slope_mean`
- `mean_cn`

### Historical Feature

- `impact`

Therefore:

    Final baseline feature count = 20

---

# 3. Target Variable

The target variable is:

    `next_month_impact`

It represents whether flood impact is expected in the following month for a given Revenue Circle.

The target was created by temporally shifting the original `impact` variable within each Revenue Circle.

Conceptually:

    Current-period information
              ↓
       Prediction period
              ↓
      Next-month impact

The model must use only information available before the target month's impact occurs.

---

# 4. Prediction Problem Type

The target is binary:

    0 → No next-month flood impact
    1 → Next-month flood impact

Therefore, this is treated as a:

    Binary Classification Problem

The model predicts the probability of next-month flood impact and converts that probability into a class prediction.

---

# 5. Class Imbalance

The target distribution identified during EDA is highly imbalanced.

The observed distribution is approximately:

| Target | Observations | Percentage |
|--------|--------------|------------|
| 0 | 10,103 | 92.01% |
| 1 | 877 | 7.99% |

Therefore, approximately 8% of observations belong to the positive flood-impact class.

This means that accuracy alone is not sufficient for evaluating the model.

A model could achieve high accuracy simply by predicting the majority class most of the time.

Therefore, the following metrics are considered:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Special attention is given to:

    Recall
    F1 Score
    ROC-AUC

because correctly identifying flood-impact cases is important for the proposed decision-support system.

---

# 6. Data Preparation for Modeling

Before model training, the following steps are performed:

1. Verify the final feature set.
2. Verify the target variable.
3. Check missing values.
4. Check data types.
5. Check target distribution.
6. Sort observations according to time.
7. Separate prediction features from identifiers.
8. Construct training and testing datasets.

The machine learning model receives only the selected 20 prediction features.

Identifiers and administrative variables are not directly used as numerical prediction features.

---

# 7. Temporal Validation Strategy

Because the project predicts future flood impact, a purely random train-test split is not preferred as the primary validation strategy.

The data contains repeated observations across Revenue Circles and time periods.

Therefore, the preferred approach is:

    Earlier time periods
             ↓
         Training
             ↓
      Later time periods
             ↓
          Testing

This better represents the real-world prediction problem.

The model is therefore evaluated on later observations that occur after the training period.

---

# 8. Training and Testing Dataset

The available observations are divided chronologically into:

    Training Dataset
          ↓
    Earlier observations

    Testing Dataset
          ↓
    Later observations

The exact number of observations and time periods used in each partition are recorded after the final dataset is loaded.

The testing data is kept separate from model fitting and is used only for final evaluation.

---

# 9. Baseline Machine Learning Models

Multiple machine learning algorithms are evaluated rather than assuming that one model will always perform best.

The baseline models are:

### 9.1 Logistic Regression

Logistic Regression provides a simple and interpretable baseline.

It helps determine whether more complex tree-based models provide a meaningful improvement.

---

### 9.2 Random Forest

Random Forest is used as a strong tree-based baseline.

It can capture nonlinear relationships and interactions between environmental, population and geographic variables.

Class weighting is used to account for the imbalanced target.

---

### 9.3 XGBoost

XGBoost is a gradient-boosting algorithm capable of learning nonlinear relationships and complex feature interactions.

It is evaluated as one of the main high-performance models.

Class imbalance is considered during training.

---

### 9.4 CatBoost

CatBoost is included as one of the primary gradient-boosting models.

It is particularly important because the proposed Stage 1 predictive engine uses tree-based boosting together with SHAP-based explainability.

CatBoost is therefore evaluated both for predictive performance and for its compatibility with later TreeSHAP analysis.

---

# 10. Model Evaluation Metrics

Each model is evaluated using:

### Accuracy

Measures the overall proportion of correctly classified observations.

However, because the target is highly imbalanced, accuracy is not considered sufficient by itself.

---

### Precision

Measures how many observations predicted as flood impact actually belong to the positive class.

---

### Recall

Measures how many actual flood-impact observations are correctly identified by the model.

Recall is particularly important because missing an actual flood-impact case can be operationally costly.

---

### F1 Score

F1 Score provides a balance between precision and recall.

It is especially useful when the classes are imbalanced.

---

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two target classes across different classification thresholds.

---

# 11. Model Comparison

The trained models are compared using a common evaluation table.

The comparison includes:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated |
| Random Forest | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated |
| XGBoost | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated |
| CatBoost | To be calculated | To be calculated | To be calculated | To be calculated | To be calculated |

The final values will be updated after model training.

The best model should not be selected using accuracy alone.

F1 Score, Recall and ROC-AUC should also be considered because the positive class represents only a small proportion of the observations.

---

# 12. Class Imbalance Handling

Because the positive class represents approximately 7.99% of observations, class imbalance is explicitly considered during training.

Depending on the model, imbalance handling includes:

- class weights
- positive-class weighting
- model-specific imbalance parameters

The purpose is to prevent the model from simply favoring the majority class.

---

# 13. Confusion Matrix Analysis

A confusion matrix is generated for the trained models, particularly for the selected final model.

The confusion matrix contains:

- True Negatives
- False Positives
- False Negatives
- True Positives

The False Negative value is particularly important for the flood-impact prediction problem because it represents an actual future-impact case that the model failed to identify.

---

# 14. Feature Importance Analysis

Tree-based models provide feature-importance information.

Feature importance is used to identify which of the 20 baseline features contribute most strongly to model predictions.

The analysis helps answer questions such as:

- How important is rainfall?
- Does historical `impact` strongly influence future impact?
- Do population variables contribute significantly?
- How important are geographical characteristics?
- Do infrastructure variables contribute to prediction?

Feature importance is treated as an initial interpretation method.

More detailed explanations will be obtained using SHAP.

---

# 15. Explainable AI — SHAP

The selected tree-based model will subsequently be analysed using SHAP (SHapley Additive exPlanations).

SHAP will be used to determine how individual features influence model predictions.

The analysis will provide:

### Global Explanation

Which features are generally the most influential across the dataset.

### Local Explanation

Why a particular Revenue Circle received a high or low predicted flood-impact probability.

Conceptually:

    Input Features
          ↓
    Trained ML Model
          ↓
    Prediction
          ↓
        SHAP
          ↓
    Feature Contributions

This forms the explainability component of Stage 1 of the proposed DSS.

---

# 16. Prediction Output

The final model will produce:

- predicted class
- predicted probability of next-month flood impact

The predicted probability is particularly useful for the later DSS stage because it can be converted into an estimated level of flood-impact need.

Conceptually:

    20 Input Features
            ↓
      Trained Model
            ↓
    Impact Probability
            ↓
    Predicted Flood Need
            ↓
    Relief Optimization

---

# 17. Relationship with the Relief Optimization Stage

The machine learning model represents Stage 1 of the proposed two-stage DSS.

### Stage 1 — Predictive Engine

    Hazard
    +
    Exposure
    +
    Vulnerability
    +
    Historical Impact
            ↓
    Machine Learning
            ↓
    Predicted Flood Impact

### Stage 2 — Prescriptive Engine

    Predicted Impact
            +
    Resource Constraints
            ↓
    Integer Linear Programming
            ↓
    Relief Allocation

The Stage 1 predictions will therefore provide information for the later resource-allocation model.

---

# 18. Important Leakage Considerations

The modelling stage follows the feature-screening decisions documented during EDA.

Flood consequences and response variables are not included as direct prediction features.

Examples of excluded variables include:

- affected population
- relief camps
- relief centre information
- camp inmates
- houses damaged
- animal damage
- crop damage
- deaths
- expenditure
- tenders
- repair and restoration

These variables describe flood consequences or government response and may not be available at the prediction time.

Using them directly could introduce target leakage.

---

# 19. Important Role of Historical Impact

The `impact` feature is intentionally retained as one of the 20 baseline features.

It represents historical/current-period flood impact and is used to predict subsequent impact.

The intended relationship is:

    Previous/current impact
            ↓
    Next-month impact

It must not become:

    Next-month impact
            ↓
    Next-month impact

Therefore, temporal ordering must be maintained during dataset construction and validation.

---

# 20. Model Selection Criteria

The final model will be selected based on a combination of:

1. F1 Score
2. Recall
3. ROC-AUC
4. Precision
5. Accuracy
6. Stability between training and testing performance
7. Suitability for SHAP explainability
8. Suitability for the later DSS pipeline

A model with slightly lower accuracy but substantially better recall and F1 Score may be preferred if it identifies more actual flood-impact cases.

---

# 21. Training vs Testing Performance

Training performance and testing performance will be compared to identify possible overfitting.

A large difference between training and testing performance may indicate:

    Overfitting

A model that performs well on both datasets is preferred.

The final analysis should therefore report both training and testing performance rather than reporting testing accuracy alone.

---

# 22. Final Modeling Results

This section will be completed after all models have been trained.

The final results will include:

- Best-performing model
- Training performance
- Testing performance
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion matrix
- Feature importance

The selected model will then be passed to the explainability stage.

---

# 23. Modeling Conclusions

The modeling stage establishes the predictive component of the Assam Flood DSS.

The current baseline uses:

    20 prediction features
             ↓
    Binary classification
             ↓
    Next-month flood impact

Multiple machine learning algorithms are evaluated to identify a suitable predictive model.

The selected tree-based model will subsequently be used for SHAP-based explanation.

The final output of Stage 1 will be a prediction of flood-impact likelihood for each Revenue Circle, which can then support the Stage 2 relief-resource optimization framework.

---

# 24. Next Steps

The next stages of the project are:

1. Finalize model comparison.
2. Select the best-performing model.
3. Analyse training vs testing performance.
4. Generate confusion matrix and performance plots.
5. Perform TreeSHAP analysis.
6. Generate global and local feature explanations.
7. Convert predicted impact into relief-demand estimates.
8. Develop the Integer Linear Programming optimization model.
9. Optimize allocation of boats, food, medical teams and other resources.
10. Integrate the predictive and optimization components into the final DSS.

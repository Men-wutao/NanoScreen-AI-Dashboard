# NanoScreen-AI Dashboard

NanoScreen-AI is a Streamlit-based machine-learning dashboard for screening-oriented prioritization of nanoparticle formulations with high tumor delivery potential.

The dashboard reformulates 24 h tumor delivery efficiency into a training-set-defined high-delivery classification task and applies a retained CatBoost model for prediction, ranking, interpretation, and candidate screening.

## Online Dashboard

The deployed dashboard is available at:

https://nanoscreen-ai-dashboard.streamlit.app

## Main Features

- Single nanoparticle formulation prediction
- Batch candidate screening from CSV or Excel files
- Top-ranked candidate formulation visualization
- Local working range recommendation
- Feature-importance-based model interpretation
- Independent test-set model evaluation
- ROC curve, precision–recall curve, score distribution, and confusion matrix visualization

## Dashboard Modules

### Overview

Summarizes the NanoScreen-AI workflow, study framework, core model results, and dashboard modules.

### Model Prediction

Allows users to input a single nanoparticle candidate formulation and estimate its probability of being classified as a high-delivery candidate.

### Candidate Screening

Allows users to upload CSV or Excel files for batch prediction, ranking, and prioritization of multiple nanoparticle candidate formulations.

### Top Candidates

Displays model-prioritized candidate formulations, including paper-reported Top 10, paper-reported Top 200, generated Top 200, and all scored candidates.

### Local Working Range

Displays model-prioritized local working ranges for key continuous variables, including Size, Zeta Potential, and Admin dose.

### SHAP Explanation

Displays feature-importance results from the retained CatBoost model and aggregates preprocessed one-hot features back to original predictor-level variables.

### Model Evaluation

Summarizes independent test-set performance using threshold-based metrics, ranking-oriented enrichment metrics, ROC/PR curves, score distribution, and confusion matrix.

## Required Input Columns for Batch Screening

Uploaded CSV or Excel files should include the following columns:

```text
Type, MAT, TS, CT, TM, Shape, Size, Zeta Potential, Admin

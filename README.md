# NanoScreen-AI Dashboard

NanoScreen-AI is an interactive Streamlit-based machine-learning dashboard for screening-oriented prioritization of nanoparticle formulations with high predicted tumor-delivery potential.

The dashboard reformulates 24 h tumor delivery efficiency into a training-set-defined Q₀.₇₅ high-delivery classification task and applies a retained CatBoost screening model for individual-formulation prediction, batch candidate screening and ranking, candidate prioritization, model interpretation, and model-predicted range estimation.

## Online Dashboard

The deployed dashboard is available at:

https://nanoscreen-ai-dashboard.streamlit.app

## Main Features

- Individual-formulation prediction
- Batch candidate screening and ranking from CSV or Excel files
- Top-ranked candidate review
- Candidate-specific model-predicted ranges
- Model interpretation using feature-importance results
- Independent test-set performance review
- ROC curve, precision–recall curve, probability distribution, confusion matrix, and ranking-oriented metric visualization

## Dashboard Modules

### Overview

Summarizes the NanoScreen-AI workflow, study framework, core model results, and dashboard modules.

### Formulation Prediction

Allows users to input a single nanoparticle formulation and estimate its probability of being classified as a high-delivery candidate using the retained CatBoost screening model.

Outputs include:

- High-delivery probability
- Predicted class
- Decision threshold
- Priority level

### Batch Screening

Allows users to upload CSV or Excel files for batch prediction, ranking, and prioritization of multiple nanoparticle candidate formulations.

Outputs include:

- Batch probabilities
- Ranked candidates
- Class distribution
- Downloadable screening results

### Top-Ranked Candidates

Displays model-prioritized nanoparticle formulations, including:

- Dataset Top 10
- Dataset Top 200
- Virtual Top 200
- Formulation details

### Model-Predicted Ranges

Displays model-prioritized ranges for key continuous formulation variables, including:

- Size
- Zeta Potential
- Admin dose

These ranges are intended to support formulation design and experimental planning and should be interpreted as model-supported exploratory windows rather than experimentally validated optimal conditions.

### Model Interpretation

Displays feature-importance results from the retained CatBoost screening model.

The module provides:

- Original-feature-level importance
- Preprocessed feature importance
- Feature-importance tables for model interpretation

Preprocessed one-hot features are aggregated back to the original predictor level to improve interpretability.

### Model Performance

Summarizes independent test-set performance of the retained CatBoost screening model using threshold-based classification metrics and ranking-oriented screening metrics.

The module includes:

- ROC curve
- Precision–Recall curve
- Probability distribution
- Confusion matrix
- Ranking-oriented screening metrics

## Required Input Columns for Batch Screening

Uploaded CSV or Excel files should include the following columns:

```text
Type, MAT, TS, CT, TM, Shape, Size, Zeta Potential, Admin

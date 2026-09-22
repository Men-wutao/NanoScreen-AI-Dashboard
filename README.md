# NanoScreen-AI Dashboard

An AI-assisted Streamlit dashboard for prioritizing nanoparticle formulations with high predicted tumor-delivery potential.

## Overview

NanoScreen-AI integrates machine learning-based screening, candidate ranking, model interpretation, and model-predicted parameter range estimation for nanoparticle formulation prioritization.

## Features

- Individual formulation prediction
- Batch candidate screening
- Virtual candidate generation and ranking
- Model interpretation using feature importance
- Candidate-specific predicted parameter ranges
- Independent test-set performance evaluation

## Model

The retained screening model is a CatBoost classifier trained to identify high-delivery nanoparticle formulations.

## Repository Structure

.
├── app.py
├── data/
├── model/
├── results/
├── figures/
└── requirements.txt


## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt

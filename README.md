# CMOR 438 Final Project: Music Analytics with a From-Scratch Machine Learning Package

## Project Overview

This repository contains a Rice University CMOR 438 / INDE 577 final project focused on music analytics with an educational, from-scratch machine learning package. The project studies how Spotify-style audio features can support:

- binary popularity classification
- unsupervised clustering into interpretable musical profiles

All core learning algorithms are implemented with `numpy` for transparency and technical depth.

## Motivating Questions

- Can audio features help predict whether a song belongs to a relatively high-popularity group?
- Do songs naturally cluster into interpretable musical profiles based on audio characteristics?

## Implemented Package Components

Current `music_ml` components include:

- `LogisticRegression` (binary, batch gradient descent, loss tracking)
- `KMeans` (from-scratch clustering with inertia and convergence tracking)
- `StandardScaler`
- `train_test_split`

Planned near-term package additions include classification metrics (`accuracy`, `precision`, `recall`, `f1`, and `confusion_matrix`) to align with the existing testing and notebook workflow.

## Repository Structure

```text
.
├── src/music_ml/
│   ├── preprocessing.py
│   ├── supervised/
│   │   └── logistic_regression.py
│   └── unsupervised/
│       └── kmeans.py
├── tests/
├── notebooks/
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── pyproject.toml
└── requirements.txt
```

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running Tests

Run the full test suite with `pytest`:

```bash
pytest
```

## Notebooks

- `notebooks/01_data_overview_and_eda.ipynb`  
  Starter EDA workflow for inspecting distributions, missingness, correlations, and popularity behavior.
- `notebooks/02_song_popularity_classification.ipynb`  
  Binary classification pipeline using custom preprocessing and custom logistic regression, with evaluation scaffolding and a scikit-learn baseline comparison.
- `notebooks/03_musical_profile_clustering.ipynb`  
  Clustering workflow using custom KMeans, elbow-style model selection, centroid inspection, PCA visualization, and scikit-learn comparison.

## Why Music Analytics?

Music analytics is a compelling machine learning application because it combines high-dimensional numerical features with intuitive human interpretation. It supports both predictive tasks (e.g., popularity grouping) and exploratory tasks (e.g., discovering musical profiles), making it a strong setting for studying model behavior, interpretability, and evaluation.

## Future Improvements

- Add robust data validation and schema checks for incoming datasets
- Expand evaluation utilities (e.g., ROC analysis and clustering quality diagnostics)
- Add notebook-to-report automation for reproducible project deliverables
- Introduce additional from-scratch models and richer hyperparameter tuning utilities

## Course Attribution

Developed as a final project for Rice University CMOR 438 / INDE 577: Data Science and Machine Learning.
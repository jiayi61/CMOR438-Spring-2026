# CMOR 438 Final Project: Music Analytics with a From-Scratch Machine Learning Package

## Overview

This repository contains a final project for Rice University CMOR 438 / INDE 577 focused on music analytics with a custom Python machine learning package. The project studies how song-level audio features can support both supervised prediction and unsupervised pattern discovery, with an emphasis on transparent, from-scratch implementations.

## Motivating Questions

- Can audio features help predict whether a song belongs to a relatively high-popularity group?
- Do songs naturally cluster into interpretable musical profiles based on audio characteristics?

## Planned Components

The `music_ml` package is being developed to include:

- Logistic Regression
- KMeans Clustering
- StandardScaler
- train_test_split
- Classification evaluation tools: accuracy, precision, recall, F1 score, and confusion matrix

## Repository Structure

The project is organized around the following directories:

- `src/music_ml` - source code for from-scratch machine learning components
- `tests` - unit tests with `pytest`
- `notebooks` - exploratory analysis, experiments, and comparison baselines
- `data` - datasets and intermediate data assets
- `figures` - generated plots and visual outputs for reporting

## Project Goals

- Build core machine learning algorithms from scratch using `numpy` for technical depth and clarity
- Maintain reproducible workflows through clean package structure, versioned code, and test coverage
- Evaluate model behavior rigorously with clear metrics and diagnostics
- Prioritize interpretability so model outputs and cluster profiles are understandable and communicable

## Course Attribution

Developed as a final project for Rice University CMOR 438 / INDE 577: Data Science and Machine Learning.
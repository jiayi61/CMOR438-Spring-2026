# CMOR 438 Final Project  
## Music Analytics with a From-Scratch Machine Learning Package

This repository contains my final project for CMOR 438 / INDE 577 at Rice University.  
The project combines from-scratch machine learning implementations with an applied analysis of a Spotify-style dataset to study patterns in song popularity and musical structure.

---

## 📌 Project Overview

This project investigates how audio features relate to song popularity and whether songs naturally group into interpretable musical profiles.

It is centered around two key questions:

1. **Can we predict whether a song is relatively popular using audio features?**  
2. **Do songs cluster into meaningful musical profiles based on their audio characteristics?**

---

## ⚙️ Custom Package: `music_ml`

The `src/music_ml` package is implemented from scratch using `numpy` and includes:

### Preprocessing
- `train_test_split`
- `StandardScaler` (with zero-variance handling)

### Supervised Learning
- `LogisticRegression`
  - Binary classification
  - Batch gradient descent optimization
  - Binary cross-entropy loss
  - Probability outputs (`predict_proba`)
  - Training loss tracking (`loss_history_`)

### Unsupervised Learning
- `KMeans`
  - Random initialization
  - Iterative centroid updates
  - Convergence detection
  - Empty cluster handling
  - Inertia (SSE) computation

### Evaluation Metrics
- accuracy
- precision
- recall
- f1 score
- confusion matrix

---

## 📊 Dataset

The analysis uses a Spotify-style dataset containing audio features such as:

- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- liveness
- valence
- tempo
- duration_ms
- popularity

Data organization:

data/
├── raw/
├── processed/

Data preprocessing is handled by:
scripts/prepare_spotify_data.py

---

## 📓 Notebooks

### 1. Data Exploration  
`notebooks/01_data_overview_and_eda.ipynb`

- Feature distributions  
- Correlation analysis  
- Initial exploration of popularity patterns  

---

### 2. Song Popularity Classification  
`notebooks/02_song_popularity_classification.ipynb`

- Binary classification (top 30% popularity vs others)  
- Custom Logistic Regression pipeline  
- Evaluation using accuracy, precision, recall, and F1  
- Feature importance analysis  

**Key Findings (data-driven):**

- `danceability` has the strongest positive coefficient  
- `loudness` also contributes positively  
- `instrumentalness` has the strongest negative effect  
- `tempo` has near-zero influence  

These results suggest that, in this dataset, songs that are more danceable and louder are more likely to fall into the high-popularity group, while instrumental-heavy tracks are less likely to be popular.

---

### 3. Musical Profile Clustering  
`notebooks/03_musical_profile_clustering.ipynb`

- Custom KMeans clustering  
- Elbow method (optimal k ≈ 4)  
- PCA visualization (for interpretation only)  
- Cluster interpretation based on centroid values  

**Key Findings:**

- Songs form distinct groups rather than a continuous distribution  
- Clusters correspond to different audio profiles  

---

## 📊 Results & Insights

Across both supervised and unsupervised methods, the results indicate that audio features contain meaningful signals:

- Popular songs tend to be **more danceable and louder**  
- Songs with higher **instrumentalness** are less likely to be popular  
- Music data naturally separates into **distinct audio-based profiles**  

These findings should be interpreted as **correlations rather than causal relationships**.

---

## 🧪 Testing

Unit tests are implemented using `pytest` and cover:

- preprocessing utilities  
- evaluation metrics  
- logistic regression  
- KMeans clustering  

Run tests with:

pytest

---

## 🚀 Installation

python3 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  

---

## 🧠 Project Highlights

- From-scratch implementations of core ML algorithms  
- Complete pipeline: data → model → evaluation → interpretation  
- Clean separation between package code and analysis notebooks  
- Reproducible and well-structured repository  

---

## ⚠️ Limitations

- Logistic regression captures only linear relationships  
- Results are dataset-specific and not causal  
- KMeans is sensitive to feature scaling and cluster selection  

---

## 🔮 Future Work

- Add additional models (e.g., KNN, tree-based methods)  
- Improve feature engineering  
- Explore deeper or nonlinear models  

---

## 📚 Course Information

Rice University  
CMOR 438 / INDE 577  
Data Science and Machine Learning

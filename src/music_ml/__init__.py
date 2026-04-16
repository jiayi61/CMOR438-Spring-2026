"""Top-level package for music_ml."""

from .metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from .preprocessing import StandardScaler, train_test_split
from .supervised import LogisticRegression
from .unsupervised import KMeans

__all__ = [
    "supervised",
    "unsupervised",
    "train_test_split",
    "StandardScaler",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "LogisticRegression",
    "KMeans",
]

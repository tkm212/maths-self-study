"""Algorithms for Deep Learning Ch. 7 (Regularization) dashboard pages."""

from __future__ import annotations

EARLY_STOPPING = (
    "Early stopping on a validation set",
    [
        "Split data into training and validation sets.",
        "Each epoch: update parameters on training loss.",
        "Track validation error after every epoch.",
        "Save parameters at the epoch with lowest validation error.",
        "Stop when validation error has not improved for $k$ consecutive epochs.",
    ],
)

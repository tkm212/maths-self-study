"""Theorems and key results for AFML Ch. 2 dashboard pages."""

from __future__ import annotations

BAR_TYPES = [
    (
        "Mandelbrot–Taylor (1967)",
        r"Price changes over a fixed number of transactions are closer to Gaussian; "
        r"over fixed time they are Paretian (heavy-tailed). Transaction-based bars improve return normality.",
    ),
]

CUSUM = [
    (
        "CUSUM vs Bollinger triggers",
        r"Unlike band-based rules, CUSUM requires a full cumulative run to exceed the threshold before firing. "
        r"Price hovering near a level does not produce a burst of redundant events.",
    ),
]

PCA_WEIGHTS = [
    (
        "Marchenko–Pastur (book context)",
        r"López de Prado (Ch. 2) uses Marchenko–Pastur denoising to separate signal eigenvalues from noise. "
        r"This demo shows plain PCA on multi-horizon returns without the denoising step.",
    ),
]

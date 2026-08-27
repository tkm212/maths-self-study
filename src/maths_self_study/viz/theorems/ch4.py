"""Theorems for Deep Learning Ch. 4 (Numerical Computation) dashboard pages."""

from __future__ import annotations

STABILITY = [
    (
        "Log-sum-exp identity",
        "For any z₁, …, zₙ, log Σᵢ exp(zᵢ) = maxᵢ zᵢ + log Σᵢ exp(zᵢ − maxⱼ zⱼ), keeping all exponentials finite.",
    ),
]

CONDITIONING = [
    (
        "Condition-number perturbation bound",
        "For Ax = b and perturbed system (A + ΔA)x' = b + Δb, relative error in x is bounded by κ(A) times relative input error, up to first order.",
    ),
]

LEAST_SQUARES = [
    (
        "Normal-equation solution",
        "If XᵀX is invertible, w* = (XᵀX)⁻¹Xᵀy is the unique minimiser of ‖Xw − y‖₂².",
    ),
]

KKT = [
    (
        "Karush-Kuhn-Tucker conditions",
        "For min f(x) s.t. g(x) ≤ 0 with smooth f, g, a feasible point x* is optimal iff "
        "∇f(x*) + λ∇g(x*) = 0, g(x*) ≤ 0, λ ≥ 0, and λ g(x*) = 0.",
    ),
]

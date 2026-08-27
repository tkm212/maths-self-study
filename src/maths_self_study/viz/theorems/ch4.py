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

GRADIENT_DESCENT = [
    (
        "Descent lemma (smooth functions)",
        "If f is L-smooth, then f(x − η∇f(x)) ≤ f(x) − η‖∇f(x)‖² + (Lη²/2)‖∇f(x)‖². "
        "Choosing η ≤ 1/L guarantees decrease when ∇f(x) ≠ 0.",
    ),
]

NEWTON = [
    (
        "Newton step on quadratics",
        "For f(x) = ½xᵀHx + bᵀx with H positive definite, one Newton step x − H⁻¹∇f(x) reaches the unique minimiser.",
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

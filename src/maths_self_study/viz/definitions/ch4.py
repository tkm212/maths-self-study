"""Definitions for Deep Learning Ch. 4 (Numerical Computation) dashboard pages."""

from __future__ import annotations

STABILITY = [
    (
        "Numerical overflow",
        "Floating-point exponents have finite range; exp(z) overflows when z is large, producing inf and breaking downstream normalisation.",
    ),
    (
        "Log-sum-exp trick",
        "Rewriting log Σ exp(zᵢ) as max(z) + log Σ exp(zᵢ − max(z)) keeps exponentials bounded and softmax numerically stable.",
    ),
]

CONDITIONING = [
    (
        "Condition number",
        "κ(A) = σ_max/σ_min ratio of largest to smallest singular value. "
        "Large κ(A) means Ax = b is sensitive to tiny changes in b or rounding error in A.",
    ),
    (
        "Ill-conditioned system",
        "When κ(A) is huge, nearly parallel rows make different inputs look alike while solutions can differ enormously.",
    ),
]

GRADIENT_DESCENT = [
    (
        "Gradient",
        "∇f(x) points in the direction of steepest ascent. "
        "First-order methods move opposite to the gradient to decrease f.",
    ),
    (
        "Learning rate",
        "Step size η in x ← x − η∇f(x). Too large causes oscillation; too small slows convergence.",
    ),
]

NEWTON = [
    (
        "Newton's method",
        "Uses second-order curvature: x ← x − H⁻¹∇f(x) where H is the Hessian. "
        "Converges fast near a minimum but costs O(n³) per step and needs H invertible.",
    ),
    (
        "Hessian",
        "Matrix of second partial derivatives H_ij = ∂²f/∂xᵢ∂xⱼ. "
        "Its eigenvalues describe local curvature along each direction.",
    ),
]

LEAST_SQUARES = [
    (
        "Linear least squares",
        "Find w minimising ‖Xw − y‖₂² — the best linear fit when equations Xw = y are overdetermined or noisy.",
    ),
    (
        "Normal equations",
        "Critical points satisfy XᵀXw = Xᵀy. "
        "When XᵀX is invertible, w = (XᵀX)⁻¹Xᵀy is the unique minimiser.",
    ),
]

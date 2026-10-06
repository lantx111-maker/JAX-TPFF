"""Constitutive maps for thermomechanical phase-field fracture."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as np

jax.config.update("jax_enable_x64", True)

safe_plus = lambda x: 0.5 * (x + np.abs(x))
safe_minus = lambda x: 0.5 * (x - np.abs(x))


@dataclass(frozen=True)
class Material:
    E: float
    nu: float
    Gc0: float
    ell: float
    residual_k: float = 1.0e-6
    thermal_residual_k: float = 1.0e-4
    plane_stress: bool = True
    model: str = "AT2"
    eta: float = 0.0
    heat_capacity: float = 0.0
    kappa0: float = 0.0
    alpha: float = 0.0
    T_ref: float = 300.0
    T_max: float = 1000.0
    b1: float = 0.0
    b2: float = 0.0
    ft: float = 0.0
    p: float = 2.0
    a2: float = 0.0
    a3: float = 0.0
    degrade_conductivity: bool = False

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lmbda(self) -> float:
        lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        if self.plane_stress:
            mu = self.mu
            return 2.0 * mu * lmbda / (lmbda + 2.0 * mu)
        return lmbda

    @property
    def c_omega(self) -> float:
        if self.model == "AT2":
            return 2.0
        if self.model in ("AT1", "AT1-Cohesive"):
            return 8.0 / 3.0
        if self.model == "PFCZM":
            return np.pi
        raise ValueError(self.model)


def gc_of_T(mat: Material, T):
    xi = (T - mat.T_ref) / mat.T_max
    return mat.Gc0 * (1.0 - mat.b1 * xi + mat.b2 * xi * xi)


def _a1(mat: Material):
    if mat.ft <= 0.0:
        return 1.0
    return 4.0 * mat.E * mat.Gc0 / (mat.c_omega * mat.ell * mat.ft ** 2)


def degradation(d, mat: Material):
    d = d[0] if d.ndim > 0 else d
    if mat.model in ("AT2", "AT1"):
        # Normalised residual stiffness: g_u(0)=1 and g_u(1)=k_u.
        return mat.residual_k + (1.0 - mat.residual_k) * (1.0 - d) ** 2
    f1 = np.abs(1.0 - d) ** mat.p
    a1 = _a1(mat)
    f2 = f1 + a1 * d + a1 * mat.a2 * d * d + a1 * mat.a3 * d ** 3
    return mat.residual_k + (1.0 - mat.residual_k) * f1 / f2


def g_prime(d, mat: Material):
    d0 = d[0] if getattr(d, "ndim", 0) > 0 else d
    if mat.model in ("AT2", "AT1"):
        return -2.0 * (1.0 - mat.residual_k) * (1.0 - d0)
    # PF-CZM: g = f1/f2, f1=(1-d)^p, f2=f1 + a1 d + a1 a2 d^2 + a1 a3 d^3
    p = mat.p
    a1 = _a1(mat)
    f1 = np.abs(1.0 - d0) ** p
    f2 = f1 + a1 * d0 + a1 * mat.a2 * d0 * d0 + a1 * mat.a3 * d0 ** 3
    f1p = -p * np.abs(1.0 - d0) ** (p - 1.0)
    f2p = f1p + a1 + 2.0 * a1 * mat.a2 * d0 + 3.0 * a1 * mat.a3 * d0 * d0
    return (1.0 - mat.residual_k) * (f1p * f2 - f1 * f2p) / (f2 * f2 + 1e-30)


def omega(d, mat: Material):
    d = d[0] if d.ndim > 0 else d
    if mat.model == "AT2":
        return d * d
    if mat.model in ("AT1", "AT1-Cohesive"):
        return d
    return 2.0 * d - d * d


def omega_prime(d, mat: Material):
    d = d[0] if d.ndim > 0 else d
    if mat.model == "AT2":
        return 2.0 * d
    if mat.model in ("AT1", "AT1-Cohesive"):
        return 1.0
    return 2.0 * (1.0 - d)


def strain(u_grad):
    return 0.5 * (u_grad + u_grad.T)


def mechanical_strain(u_grad, T, mat: Material):
    return strain(u_grad) - mat.alpha * (T - mat.T_ref) * np.eye(u_grad.shape[0])


def psi_plus_of_epsilon(epsilon, lmbda, mu):
    evals = np.linalg.eigh(epsilon)[0]
    tr_plus = safe_plus(np.trace(epsilon))
    return 0.5 * lmbda * tr_plus ** 2 + mu * np.sum(safe_plus(evals) ** 2)


def psi_minus_of_epsilon(epsilon, lmbda, mu):
    evals = np.linalg.eigh(epsilon)[0]
    tr_minus = safe_minus(np.trace(epsilon))
    return 0.5 * lmbda * tr_minus ** 2 + mu * np.sum(safe_minus(evals) ** 2)


def make_stress_fn(mat: Material, dim: int, key=0):
    lmbda, mu = mat.lmbda, mat.mu
    rng = jax.random.PRNGKey(key)
    noise = jax.random.uniform(rng, shape=(dim, dim), minval=-1e-10, maxval=1e-10)
    noise = np.diag(np.diag(noise))

    def stress_fn(u_grad, d, T=None):
        T_val = mat.T_ref if T is None else np.reshape(T, ())
        eps = mechanical_strain(u_grad, T_val, mat) + noise
        g = degradation(d, mat)
        return g * jax.grad(lambda e: psi_plus_of_epsilon(e, lmbda, mu))(eps) + jax.grad(
            lambda e: psi_minus_of_epsilon(e, lmbda, mu)
        )(eps)

    def psi_plus_fn(u_grad, T=None):
        T_val = mat.T_ref if T is None else np.reshape(T, ())
        eps = mechanical_strain(u_grad, T_val, mat)
        return psi_plus_of_epsilon(eps, lmbda, mu)

    return stress_fn, psi_plus_fn

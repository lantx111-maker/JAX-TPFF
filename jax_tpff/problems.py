"""JAX-FEM problems for displacement, damage and temperature."""

from __future__ import annotations

import jax_tpff.jax_compat  # noqa: F401
import jax
import jax.numpy as np
import numpy as onp

from jax_fem.problem import Problem

from jax_tpff.constitutive import Material, gc_of_T, g_prime, make_stress_fn, omega_prime

jax.config.update("jax_enable_x64", True)


class Elasticity(Problem):
    def custom_init(self, mat: Material):
        self.mat = mat
        self.stress_fn, self.psi_plus_fn = make_stress_fn(mat, self.dim)

    def get_tensor_map(self):
        stress_fn = self.stress_fn
        thermo = self.mat.alpha != 0.0

        def fn(u_grad, d, T=None):
            if thermo:
                return stress_fn(u_grad, d, T)
            return stress_fn(u_grad, d)

        return fn

    def set_params(self, params):
        if len(params) == 2:
            sol_d, disp = params
            d = self.fes[0].convert_from_dof_to_quad(sol_d)
            self.internal_vars = [d]
        else:
            sol_d, sol_T, disp = params
            d = self.fes[0].convert_from_dof_to_quad(sol_d)
            T = self.fes[0].convert_from_dof_to_quad(sol_T)
            self.internal_vars = [d, T]
        if disp is not None:
            self._set_disp(disp)

    def _set_disp(self, disp):
        pass

    def compute_history(self, sol_u, history_old, sol_T=None):
        fe = self.fes[0]
        u_grads = np.take(sol_u, fe.cells, axis=0)[:, None, :, :, None] * fe.shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)
        if sol_T is None:
            psi = jax.vmap(jax.vmap(self.psi_plus_fn))(u_grads)
        else:
            Tq = fe.convert_from_dof_to_quad(sol_T)
            psi = jax.vmap(jax.vmap(lambda g, t: self.psi_plus_fn(g, t)))(u_grads, Tq)
        return np.maximum(psi, history_old)

    def compute_psi_plus(self, sol_u, sol_T=None):
        fe = self.fes[0]
        zeros = np.zeros((fe.num_cells, fe.num_quads))
        return self.compute_history(sol_u, zeros, sol_T)

    def compute_stress_quadrature(self, sol_u, sol_d, sol_T=None):
        """Return the Cauchy stress at volume quadrature points."""
        fe = self.fes[0]
        u_grads = np.take(sol_u, fe.cells, axis=0)[:, None, :, :, None] * fe.shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)
        d_q = fe.convert_from_dof_to_quad(sol_d)
        stress = self.get_tensor_map()
        if sol_T is None:
            return jax.vmap(jax.vmap(stress))(u_grads, d_q)
        T_q = fe.convert_from_dof_to_quad(sol_T)
        return jax.vmap(jax.vmap(stress))(u_grads, d_q, T_q)

    def compute_traction(self, location_fn, sol_u, sol_d, sol_T=None, normal=None):
        fe = self.fes[0]
        stress = self.get_tensor_map()
        if normal is None:
            normal = np.array([0.0, 1.0] if self.dim == 2 else [0.0, 1.0, 0.0])

        def traction_fn(u_grads, d_face, T_face=None):
            ug = u_grads.reshape(-1, fe.vec, self.dim)
            dg = d_face.reshape(-1, d_face.shape[-1])
            if T_face is None:
                sigmas = jax.vmap(stress)(ug, dg).reshape(u_grads.shape)
            else:
                tg = T_face.reshape(-1, T_face.shape[-1])
                sigmas = jax.vmap(stress)(ug, dg, tg).reshape(u_grads.shape)
            return (sigmas @ normal[None, None, :, None])[:, :, :, 0]

        boundary_inds = fe.get_boundary_conditions_inds([location_fn])[0]
        face_shape_grads_physical, nanson_scale = fe.get_face_shape_grads(boundary_inds)
        u_grads_face = sol_u[fe.cells][boundary_inds[:, 0]][:, None, :, :, None] * face_shape_grads_physical[:, :, :, None, :]
        u_grads_face = np.sum(u_grads_face, axis=2)
        selected_cell_sols_d = sol_d[fe.cells][boundary_inds[:, 0]]
        selected_face_shape_vals = fe.face_shape_vals[boundary_inds[:, 1]]
        d_face = np.sum(selected_cell_sols_d[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2)
        T_face = None
        if sol_T is not None:
            selected_cell_sols_T = sol_T[fe.cells][boundary_inds[:, 0]]
            T_face = np.sum(selected_cell_sols_T[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2)
        traction = traction_fn(u_grads_face, d_face, T_face)
        return np.sum(traction * nanson_scale[:, :, None], axis=(0, 1))


class PhaseField(Problem):
    def custom_init(self, mat: Material):
        self.mat = mat

    def get_tensor_map(self):
        mat = self.mat
        c = mat.c_omega
        thermo = (mat.b1 != 0.0) or (mat.b2 != 0.0)

        def fn(d_grad, history, T=None):
            Gc = mat.Gc0 if not thermo else gc_of_T(mat, np.reshape(T, ()))
            return (Gc * mat.ell / c) * 2.0 * d_grad

        return fn

    def get_mass_map(self):
        mat = self.mat
        c = mat.c_omega
        thermo = (mat.b1 != 0.0) or (mat.b2 != 0.0)

        def fn(d, x, history, T=None):
            Gc = mat.Gc0 if not thermo else gc_of_T(mat, np.reshape(T, ()))
            d0 = d[0]
            hist = history[0] if getattr(history, "ndim", 0) > 0 else history
            drive = g_prime(d, mat) * hist
            geo = Gc / (c * mat.ell) * omega_prime(d, mat)
            return np.array([geo + drive])

        return fn

    def set_params(self, params):
        if len(params) == 1:
            self.internal_vars = [params[0]]
        else:
            history, sol_T = params
            T = self.fes[0].convert_from_dof_to_quad(sol_T)
            self.internal_vars = [history, T]


class Heat(Problem):
    def custom_init(
        self,
        mat: Material,
        h_conv: float = 0.0,
        T_ambient: float | None = None,
        lump_heat_capacity: bool = False,
    ):
        self.mat = mat
        self.h_conv = h_conv
        self.T_ambient = mat.T_ref if T_ambient is None else T_ambient
        self.lump_heat_capacity = lump_heat_capacity

    def get_surface_maps(self):
        h = self.h_conv
        T_inf = self.T_ambient

        def robin(T, _x):
            return h * (T - T_inf)

        return [robin for _ in self.boundary_inds_list]

    def get_tensor_map(self):
        mat = self.mat

        def fn(T_grad, T_old, dt, d=None):
            kappa = mat.kappa0
            if mat.degrade_conductivity and d is not None:
                # Normalised so that kappa(0)=kappa0 and
                # kappa(1)=thermal_residual_k*kappa0.
                g = mat.thermal_residual_k + (1.0 - mat.thermal_residual_k) * (1.0 - d[0]) ** 2
                kappa = g * mat.kappa0
            return kappa * T_grad

        return fn

    def get_mass_map(self):
        mat = self.mat

        def fn(T, x, T_old, dt, d=None):
            if self.lump_heat_capacity:
                return np.zeros_like(T)
            return mat.heat_capacity / dt * (T - T_old)

        return fn

    def get_universal_kernel(self):
        """Add a row-sum-lumped heat-capacity term when requested.

        For the first-order elements used here, row-sum lumping assigns an
        equal share of the cell volume to every node.  The diffusion and
        Robin terms remain quadrature based.  This diagonal capacity matrix
        removes the positive off-diagonal entries of the consistent mass
        matrix that caused temperature overshoots at small time steps.
        """

        mat = self.mat

        def kernel(cell_sol_flat, _x, _shape_grads, cell_JxW, _v_grads_JxW, T_old, dt, d=None):
            if not self.lump_heat_capacity:
                return np.zeros_like(cell_sol_flat)
            cell_sol = self.unflatten_fn_dof(cell_sol_flat)[0]
            n_node = cell_sol.shape[0]
            volume = np.sum(cell_JxW[0])
            dt_value = np.reshape(dt, (-1,))[0]
            old = np.reshape(T_old, cell_sol.shape)
            value = mat.heat_capacity * volume / (dt_value * n_node) * (cell_sol - old)
            return value.reshape(-1)

        return kernel

    def set_params(self, params):
        sol_T_old, dt, sol_d = params
        if self.lump_heat_capacity:
            T_old = sol_T_old[self.fes[0].cells]
            dt_cells = dt * np.ones((self.num_cells, self.fes[0].num_nodes, 1))
        else:
            T_old = self.fes[0].convert_from_dof_to_quad(sol_T_old)
            dt_cells = dt * np.ones((self.num_cells, self.fes[0].num_quads, 1))
        if sol_d is None:
            self.internal_vars = [T_old, dt_cells]
        else:
            d = self.fes[0].convert_from_dof_to_quad(sol_d)
            self.internal_vars = [T_old, dt_cells, d]

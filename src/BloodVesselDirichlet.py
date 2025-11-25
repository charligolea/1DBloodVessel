import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.sparse import diags,eye
from scipy.sparse.linalg import spsolve
from scipy.optimize import root

class Grid1D:
    """
    Create a 1D primal grid over [0, L] with optional geometric stretching.
    If stretch_factor = 1.0, generate a uniform grid.
    Returns:
        x_faces: coordinates of cell boundaries (Np + 1)
        x_centers: barycenters of each cell (Np)
        dx_primal: cell widths (Np)
    """

    def __init__(self, L, Np, stretch_factor=1.0, stretch_type='geometric', periodic=False):
        self.L = L
        self.Np = Np
        self.stretch_factor = stretch_factor

        if stretch_type == 'geometric':
            self.x_faces, self.x_centers, self.dx_primal = self._create_nonuniform_grid()
        elif stretch_type == 'uniform':
            self.stretch_factor = 1.0
            self.x_faces, self.x_centers, self.dx_primal = self._create_nonuniform_grid()
        elif stretch_type == 'chebysev':
            self.x_faces, self.x_centers, self.dx_primal = self._create_chebysev_grid()
        elif stretch_type == 'test':
            self.x_faces, self.x_centers, self.dx_primal = self._create_test_case()
        else:
            raise ValueError("stretch_type must be 'geometric', 'uniform', or 'chebysev'")

        self.x_dual_edges, self.x_dual_centers, self.dx_dual = self._create_dual_grid()

    def _create_test_case(self):
        x_faces = np.array([0, 1, 3, 6, 8, 9])
        dx_primal = np.diff(x_faces)
        self.L = x_faces[-1]
        x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
        return x_faces, x_centers, dx_primal

    def _create_chebysev_grid(self):
        # Chebyshev grid generation
        x_faces = 0.5 * self.L * (1 - np.cos(np.pi * np.arange(self.Np + 1) / self.Np))
        dx_primal = np.diff(x_faces)
        x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
        return x_faces, x_centers, dx_primal

    def _create_nonuniform_grid(self):
        if self.stretch_factor == 1.0:
            dx_primal = np.full(self.Np, self.L / self.Np)
        else:
            r = self.stretch_factor
            dx0 = self.L * (1 - r) / (1 - r ** self.Np)
            dx_primal = dx0 * r ** np.arange(self.Np)

        x_faces = np.zeros(self.Np + 1)
        x_faces[0] = 0.0
        x_faces[1:] = np.cumsum(dx_primal)

        x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
        return x_faces, x_centers, dx_primal

    def _create_dual_grid(self):
        """
        Create dual grid edges from barycenters of the primal grid.
        Dual cells lie between primal barycenters.
        """
        Nd = self.Np + 1
        x_dual_edges = np.zeros(Nd + 1)
        x_dual_edges[1:-1] = self.x_centers

        x_dual_edges[0] = 0
        x_dual_edges[-1] = self.L

        x_dual_centers = 0.5 * (x_dual_edges[:-1] + x_dual_edges[1:])
        x_dual_centers[0] = x_dual_edges[0]
        x_dual_centers[-1] = x_dual_edges[-1]

        dx_dual = np.zeros(Nd)
        dx_dual[0] = self.x_centers[0]
        dx_dual[-1] = self.L - self.x_centers[-1]
        dx_dual[1:-1] = 0.5 * (self.dx_primal[:-1] + self.dx_primal[1:])

        return x_dual_edges, x_dual_centers, dx_dual


class BloodVessel1D:
    def __init__(self, grid: Grid1D, flux_type='kolgan', BC_type="dirichlet", BC_u=[1, -0.2],
                 BC_A=[1.6 * 3.1416e-4, 1.05 * 3.1416e-4], BC_p=[0, 0], dt=1e-4, tF=0.1, rho=1050, beta=0.0, Tau=0.0,
                 A0=3.1416e-4, pe=0.0, m_param=0.5, n_param=0, K=80, nu=0):

        self.grid = grid
        self.A0 = A0
        self.dt = dt
        self.tF = tF

        self.rho = rho  # density
        self.beta = beta  #
        self.Tau = Tau  # Viscoelastic parameter
        self.pe = pe  # external pressure
        self.K = K  # stiffness coefficient

        # Behaviour of vessel wall
        # Default values are for arteries
        self.m_param = m_param
        self.n_param = n_param

        self.xc = self.grid.x_centers
        self.xd = self.grid.x_dual_centers
        self.dx_dual = grid.dx_dual
        self.dx_primal = grid.dx_primal

        self.A = np.full(len(self.xc), A0)
        self.nu = nu

        self.n = self.grid.Np

        self.gamma = np.zeros(len(self.xd))
        self.A_dual = np.ones_like(self.xd)
        self.diffusive_calc_dual_area()

        # Initial conditions
        self.q = np.ones(len(self.xd))
        self.u = np.ones_like(self.q)
        self.p = np.ones(len(self.xc))

        if BC_type == "dirichlet":
            self.p[0], self.p[-1] = BC_p[0], BC_p[1]
            self.u[0], self.u[-1] = BC_u[0], BC_u[1]
            self.A_dual[0], self.A_dual[-1] = BC_A[0], BC_A[1]
            self.q[0], self.q[-1] = self.A[0] * self.u[0], self.A[-1] * self.u[-1]
        elif BC_type == "test_1":
            self.q = self.xd
            self.A_dual = self.A0 * self.A_dual
            self.u = self.q / self.A_dual

        self.q_starstar = np.copy(self.q)
        self.q_star = np.copy(self.q)

        # Choose flux computation
        if flux_type not in ['kolgan', 'ducros']:
            raise ValueError("flux_type must be 'kolgan' or 'ducros'")
        elif flux_type == 'kolgan':
            self.flux_func = self.convective_compute_flux_kolgan
        else:
            self.flux_func = self.convective_compute_flux_ducros

        self.flux_type = flux_type

    ####### CONVECTIVE STAGE #######

    ## The first stage of the algorithm involves the spatial discretization of the equation employing an explicit finite volume method
    # based on the Ducros or Kolgan-type numerical flux function

    @staticmethod
    def minmod(a, b):
        return np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

    def convective_compute_slopes(self, q, i_start, i_end):

        if i_start < 1:
            # f_i dh_minus gets to the left edge of the grid -1/2
            diffs = np.append(q[i_start + 1] - q[i_start], np.diff(q[i_start:i_end + 1]))
        elif i_end >= len(q) - 1:
            # f_i dh_plus gets to the right edge of the grid Np + 3/2
            diffs = np.append(np.diff(q[i_start - 1:i_end]), q[i_end] - q[i_end - 1])
        else:
            diffs = np.diff(q[i_start - 1:i_end + 1])
        dq_minus = diffs[:-1]
        dq_plus = diffs[1:]

        return self.minmod(dq_minus, dq_plus)

    def convective_kolgan_reconstruct(self, h, i_start, i_end):
        # Equation 20
        # h can be q or u
        dh_minus = self.convective_compute_slopes(h, i_start - 1, i_end - 1)
        dh_plus = self.convective_compute_slopes(h, i_start, i_end)

        h_R = h[i_start - 1:i_end - 1] + (self.xc[i_start - 1:i_end - 1] - self.xd[i_start - 1:i_end - 1]) * dh_minus
        h_L = h[i_start:i_end] + (self.xc[i_start - 1:i_end - 1] - self.xd[i_start:i_end]) * dh_plus
        return h_R, h_L

    def convective_compute_flux_kolgan(self, i_start, i_end):
        # Equation 22
        q_R, q_L = self.convective_kolgan_reconstruct(self.q, i_start, i_end)
        u_R, u_L = self.convective_kolgan_reconstruct(self.u, i_start, i_end)
        a = 2 * np.maximum(np.abs(u_L), np.abs(u_R))
        return 0.5 * (self.u[i_start:i_end] + self.u[i_start - 1:i_end - 1]) * \
            (self.q[i_start:i_end] + self.q[i_start - 1:i_end - 1]) - 0.5 * a * (q_L - q_R)

    def convective_compute_flux_ducros(self, i_start, i_end):
        # Ducros flux function Eq
        a = 2 * np.maximum(np.abs(self.u), np.abs(np.roll(self.u, 1)))

        return 0.5 * (self.u[i_start:i_end] + self.u[i_start - 1:i_end - 1]) * \
            0.5 * (self.q[i_start:i_end] + self.q[i_start - 1:i_end - 1]) - \
            0.5 * a[i_start - 1:i_end - 1] * (self.q[i_start:i_end] - self.q[i_start - 1:i_end - 1])

    def convective_update_q(self):

        dx = self.dx_dual

        mask = self.u != 0

        # Select only corresponding dx and u values where u != 0
        dx_nonzero = dx[mask]
        u_nonzero = self.u[mask]
        self.q_star = np.copy(self.q)

        if u_nonzero.size > 0:
            CFL_cond = 0.5 * np.min(dx_nonzero / np.abs(u_nonzero))  # ReviseCFL condition

        if self.dt > CFL_cond:
            print("delta t is too large and will lead to unstable scheme")

        f_ip1 = self.flux_func(2, self.n + 1)
        f_i = self.flux_func(1, self.n)

        self.q_star[1:-1] -= self.dt / (self.dx_primal[1:]) * (f_ip1 - f_i)

    ####### DIFFUSIVE STAGE #######

    ## Once the first intermediate solution of the conservative variable 𝑞∗ is obtained within the convective stage, we can solve the diffusive stage following an implicit finite volume approach.
    # The discretization of the diffusive equation based on the backward in time centered in space scheme (BTCS)

    def diffusive_calc_dual_area(self):
        # Se conoce ya el area en el instante n que se ha estimado en el pressure stage n-1
        # Se hace la media aritmetica de las celdas vecinas
        self.A_dual[1:-1] = (self.A[:-1] + self.A[1:]) / 2

    def calc_phi(self, A_value):
        return self.Tau / (self.A0 * np.sqrt(A_value[1:self.n - 1]))

    def diffusive_update_q(self):
        N = self.n

        # Preallocate tridiagonal arrays
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)

        # Dual areas
        A_minus = self.A_dual[:-2]
        A_i = self.A_dual[1:-1]
        A_plus = self.A_dual[2:]

        # dx
        dxa = self.xc[1:N - 1] - self.xc[0:N - 2]
        dxb = self.xc[2:N] - self.xc[1:N - 1]
        dxc = self.xd[1:N - 1] - self.xd[0:N - 2]

        # Phi values
        phi_m = self.calc_phi(A_minus)
        phi_i = self.calc_phi(A_i)
        phi_p = self.calc_phi(A_plus)

        phi_minus = 0.5 * (phi_i + phi_m)  # centered at i-1/2
        phi_plus = 0.5 * (phi_i + phi_p)  # centered at i+1/2

        if self.nu != 0.0:
            print("we need to figure gamma out")
            a[1:N - 1] = -self.dt / self.rho * phi_minus / dxa
            b[1:N - 1] = (dxc / A_i[1:N - 1] + (1 - self.beta) * self.dt * self.gamma[1:N - 1] * dxc / A_i[1:N - 1] +
                          self.dt / self.rho * (phi_plus / dxb + phi_minus / dxa))
            c[1:N - 1] = -self.dt / self.rho * phi_plus / dxb
        else:
            a[1:N - 1] = -self.dt / self.rho * phi_minus / dxa
            c[1:N - 1] = -self.dt / self.rho * phi_plus / dxb
            b[1:N - 1] = dxc / A_i[1:N - 1] - (a[1:N - 1] + b[1:N - 1])

        # Sparse matrix and RHS
        diagonals = [
            a[2:N - 1],  # Lower diagonal (i-1)
            b[1:N - 1],  # Main diagonal (i)
            c[1:N - 2],  # Upper diagonal (i+1)
        ]
        offsets = [-1, 0, 1]
        D = diags(diagonals, offsets, shape=(N - 2, N - 2)).tocsc()

        rhs = self.q_star[1:N - 1]
        q_starstar = spsolve(D, rhs)

        # Restore boundaries
        self.q_starstar[1:N - 1] = q_starstar

    ####### PRESSURE STAGE #######

    def pressure_stage_updates(self):

        # i goes for 1 to Np in paper 0 to Np -1 in python

        N = self.n

        # 1. UPDATE PRESSURE: Solve pressure problem

        # 1.A: Compute T matrix

        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        b_rhs = np.zeros(N)

        dxa = self.dx_dual[: - 1]
        dxb = self.dx_dual[1:]

        A_minus = self.A_dual[0:-1]
        A_i = self.A_dual[1:]

        self.p_tilde = np.zeros(N)
        self.p_tilde = self.pe + self.K * ((self.A / self.A0) ** self.m_param - (self.A / self.A0) ** self.n_param)

        if self.nu != 0.0:
            print("we need to figure gamma out")
            denom_plus = self.rho * (1 + self.beta * self.dt * self.gamma[1:])
            denom_minus = self.rho * (1 + self.beta * self.dt * self.gamma[0:-1])
        else:
            denom_plus = self.rho * np.ones(N)
            denom_minus = self.rho * np.ones(N)

        a = -(self.dt) ** 2 / (dxa * denom_minus) * A_minus
        c = -(self.dt) ** 2 / (dxb * denom_plus) * A_i
        b = - (a + c) + self.A0 * self.dx_primal * (2 / self.K - 2 * self.pe / (self.K ** 2))

        # Sparse matrix and RHS
        diagonals = [
            a[1:],  # Lower diagonal (i-1)
            b,  # Main diagonal (i)
            c[:-1],  # Upper diagonal (i+1)
        ]
        offsets = [-1, 0, 1]
        T = diags(diagonals, offsets, shape=(N, N)).tocsc()
        M = self.A * self.dx_primal

        # 1.B Compute RHS vector b

        b_rhs[1:-1] = M[1:-1] - self.dt * self.rho * (
                    self.q_starstar[2:-1] / denom_plus[1:-1] - self.q_starstar[1:-2] / denom_minus[1:-1]) \
                      - self.dt ** 2 * \
                      ((1 / dxb[1:-1]) * (A_i[1:-1] / (self.rho * denom_plus[1:-1])) * (
                                  (self.p[2:] - self.p_tilde[2:]) - (self.p[1:-1] - self.p_tilde[1:-1])) \
                       - (1 / dxa[1:-1]) * (A_minus[1:-1] / (self.rho * denom_minus[1:-1])) * (
                                   (self.p[1:-1] - self.p_tilde[1:-1]) - (
                                       self.p[0:-2] - self.p_tilde[0:-2]))) - self.A0 * self.dx_primal[1:-1] * (
                                  1 - self.pe / self.K) ** 2
        b_rhs[0] = M[0] - self.dt * self.rho * (
                    self.q_starstar[1] / denom_plus[0] - self.q_starstar[0] / denom_minus[0]) \
                   - self.dt ** 2 * \
                   ((1 / dxb[0]) * (A_i[0] / (self.rho * denom_plus[0])) * (
                               (self.p[1] - self.p_tilde[1]) - (self.p[0] - self.p_tilde[0]))) - self.A0 * \
                   self.dx_primal[0] * (1 - self.pe / self.K) ** 2
        # - (1/dxa[0]) * (A_minus[0]/(self.rho*denom_minus[0]))  * ((self.p[0]-self.p_tilde[0]) - "if periodic" (self.p[-1]-self.p_tilde[-1])))
        # maybe half of this later temr is needed
        b_rhs[-1] = M[-1] - self.dt * self.rho * (
                    self.q_starstar[-1] / denom_plus[-1] - self.q_starstar[-2] / denom_minus[-1]) \
                    - self.dt ** 2 * \
                    (- (1 / dxa[-1]) * (A_minus[-1] / (self.rho * denom_minus[-1])) * (
                                (self.p[-1] - self.p_tilde[-1]) - (self.p[-2] - self.p_tilde[-2]))) - self.A0 * \
                    self.dx_primal[-1] * (1 - self.pe / self.K) ** 2
        # ((1/dxb[-1]) * (A_i[-1]/(self.rho*denom_plus[-1])) * ( - (self.p[-1]-self.p_tilde[-1]) "if peridic" : + (self.p[0]-self.p_tilde[0])) \
        # maybe half of this later temr is needed also peridoic is there

        P2 = self.A0 * self.dx_primal / (self.K ** 2) * eye(N)

        def newton_identity(p):
            return P2 @ p ** 2 + T @ p - b_rhs

        p_new = root(newton_identity, self.p, method='hybr')
        return np.array(p_new.x)

    ##### CORRECTION STAGE #####

    def correction_stage(self, p_updated):

        # 1. Convective, diffusive and pressure stages have been completed thus far

        N = self.n
        A_i = self.A_dual
        dx = self.dx_dual

        q_new = np.copy(self.q)

        # 2. UPDATE Q

        if self.nu != 0.0:
            print("We need to figure gamma out")
            q_new[1:-1] = (1 / (self.beta + self.dt * self.gamma[1:-1])) * \
                          (self.q_starstar[1:-1] \
                           - ((self.dt * A_i[1:-1]) / (self.rho * dx[1:-1])) * (p_updated[1:] - p_updated[:-1]) \
                           + ((self.dt * A_i[1:-1]) / (self.rho * dx[1:-1])) * (
                                       (self.p[1:] - self.p_tilde[1:]) - (self.p[:-1] - self.p_tilde[:-1])))
        else:
            q_new[1:-1] = self.q_starstar[1:-1] \
                          - ((self.dt * A_i[1:-1]) / (self.rho * dx[1:-1])) * (p_updated[1:] - p_updated[:-1]) \
                          + ((self.dt * A_i[1:-1]) / (self.rho * dx[1:-1])) * (
                                      (self.p[1:] - self.p_tilde[1:]) - (self.p[:-1] - self.p_tilde[:-1]))

        self.q = q_new

        # 3. UPDATE AREA
        # In case an elastic artery is considered, i.e. for 𝑚 = 0.5, 𝑛 = 0 and Γ = 0
        A_new = np.copy(self.A)
        A_new = self.A0 * (1 + (self.p - self.pe) / self.K) ** 2

        return q_new, A_new

    def test_convection(self):
        n_steps = int(self.tF / self.dt)
        q_solutions = []
        q_star_solutions = []
        print(self.q)
        q_star_solutions.append(self.q_star)
        i = 0
        t = 0
        while t < self.tF:

            self.convective_update_q()
            if i % 10 == 0:
                q_star_solutions.append(self.q_star)
            t += self.dt

            self.q = np.copy(self.q_star)
            self.q[-1] = self.xd[-1] * self.A0 / (self.A0 + 2 * t)
            i += 1

            # u_solutions[t_step, :] = q_new/A_new
        return q_star_solutions

    def solve_blood_vessel(self):
        n_steps = int(self.tF / self.dt)

        q_solutions = np.zeros((n_steps + 1, len(self.xd)))
        u_solutions = np.zeros((n_steps + 1, len(self.xd)))
        p_solutions = np.zeros((n_steps + 1, len(self.xc)))
        A_solutions = np.zeros((n_steps + 1, len(self.xc)))
        q_starstar_solutions = np.zeros((n_steps + 1, len(self.xd)))
        q_star_solutions = np.zeros((n_steps + 1, len(self.xd)))

        for t_step in range(n_steps + 1):
            self.convective_update_q()
            self.diffusive_update_q()
            p_new = self.pressure_stage_updates()
            q_new, A_new = self.correction_stage(p_new)
            self.A = A_new
            self.diffusive_calc_dual_area()

            q_solutions[t_step, :] = q_new
            A_solutions[t_step, :] = A_new
            p_solutions[t_step, :] = p_new

            q_starstar_solutions[t_step, :] = self.q_starstar
            q_star_solutions[t_step, :] = self.q_star

            # u_solutions[t_step, :] = q_new/A_new

        return q_solutions, A_solutions, p_solutions, q_star_solutions, q_starstar_solutions
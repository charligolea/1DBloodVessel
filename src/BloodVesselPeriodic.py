import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.sparse import diags,eye
from scipy.sparse.linalg import spsolve, bicg, cg
from scipy.optimize import root

PI = np.pi

class Grid1D:
    """
    Create a 1D primal grid over [0, L] with optional geometric stretching.
    If stretch_factor = 1.0, generate a uniform grid.
    Returns:
        x_faces: coordinates of primal cell boundaries (Np + 1)
        x_centers: barycenters of each cell (Np)
        dx_primal: cell widths (Np)
        
        x_dual_edges: coordinates of dual cell boundaries
        x_dual_centers: barycenters of each dual cell
        dx_dual: dual cell widths
    """

    def __init__(self, L, Np, stretch_factor=1.0, stretch_type='geometric', periodic=False):
        self.L = L
        self.Np = Np
        self.stretch_factor = stretch_factor

        if stretch_type == 'geometric':
            # Allows for non 1 stretch factor
            self.x_faces, self.x_centers, self.dx_primal = self._create_nonuniform_grid()
        elif stretch_type == 'uniform':
            self.stretch_factor = 1.0
            self.x_faces, self.x_centers, self.dx_primal = self._create_nonuniform_grid()
        elif stretch_type == 'chebysev':
            # Chebyshev point distribution
            self.x_faces, self.x_centers, self.dx_primal = self._create_chebysev_grid()
        elif stretch_type == 'test':
            # Simple test grid to test that it's being correctly done
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
        x_faces = 0.5 * self.L * (1 - np.cos(PI * np.arange(self.Np + 1) / self.Np))
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



class BloodVessel1Dperiodic:
    def __init__(self, grid: Grid1D, flux_type='kolgan', BC_type="dirichlet", dt=1e-4, tF=0.1, rho=1050, beta=1.0, Tau=0.0,
                 A0=3.1416e-4, pe=0.0, m_param=0.5, n_param=0, K=80, nu=0, xi = 0.0,  cell_averages=True, theta = 0.5):

        self.grid = grid # inherits grid properties
        self.A0 = A0 # Initial area
        self.dt_initial = dt # time step initial
        self.dt = dt # time step (may be updated with CFL condition)
        self.tF = tF # final time step

        self.rho = rho  # density
        self.beta = beta  # Factor to determine whether source term is included in diffusive or pressure stage
        self.Tau = Tau  # Viscoelastic parameter
        self.pe = pe  # external pressure
        self.K = K  # stiffness coefficient

        # Behaviour of vessel wall
        # Default values are for arteries
        self.m_param = m_param # for tube law
        self.n_param = n_param # for tube law

        # Grid parameters
        self.xc = self.grid.x_centers
        self.xd = self.grid.x_dual_centers[1:]
        self.dx_dual = grid.dx_dual
        self.dx_primal = grid.dx_primal
        
        self.A = np.full(len(self.xc), A0) # Area in primal cells
        self.nu = 0 # Viscosity
        self.s = nu * PI * xi # viscous source term, will be 0

        self.n = self.grid.Np # grid size
        self.t = 0.0 # time

        # Initialize various quantities
        self.dx_dual[-1] += self.dx_dual[0] # periodic
        self.dx_dual = self.dx_dual[1:] #remove i = 1/2
        self.gamma = np.zeros(len(self.xd)) # vicous term for pressure stage, will be null
        self.A_dual = np.ones_like(self.xd) # Area in dual cells
        self.diffusive_calc_dual_area() # Initialize area in dual cells from area in primal cells
        self.q = np.ones(len(self.xd)) # Mass flow solution array
        self.u = np.ones_like(self.q) # Axial velocity solution array
        self.p = np.ones(len(self.xc)) # Pressure solution array
        self.forcing_term_conv = np.zeros(len(self.xd))  # Forcing term for convection
        self.forcing_term_diff = np.zeros(len(self.xd))  # Forcing term for diffusion
        self.forcing_term_press = np.zeros(len(self.xd))  # Forcing term for pressure
        self.cell_averages = cell_averages
        num = n_param*(n_param - 1) # For tube law
        den = m_param*(m_param - 1)
        den_2 = m_param - n_param
        
        # For nested Newton
        self.pv = pe + K * ((num/den)**(m_param/den_2)-(num/den)**(n_param/den_2))
        self.artery_Tau0 = True
            # Initial conditions
        self.theta = theta

        # Choose flux computation: kolgan or ducros
        if flux_type not in ['kolgan', 'ducros']:
            raise ValueError("flux_type must be 'kolgan' or 'ducros'")
        elif flux_type == 'kolgan':
            self.flux_func = self.convective_compute_flux_kolgan
        else:
            self.flux_func = self.convective_compute_flux_ducros

        # Update forcing terms
        self.update_forcing = self.update_forcing_zeros

        # Set up validation tests
        if BC_type == "test_1":
            self.q = np.sin(2*PI*self.xd/grid.L)
            self.A_dual = self.A0 * self.A_dual

        elif BC_type == "test_stationary":
            self.q = np.ones(len(self.xd))
            def integrate_A(x):
                integral = (np.arctan(np.sqrt(15)*(4 * np.tan(PI * x) +1)/15 )+ PI * np.floor(x - 1/2) )/(np.sqrt(15) * PI)

                return integral

            A = 1/(np.sin(2*PI*self.xc) + 4)
            self.A = (integrate_A(self.xd) - integrate_A(self.xd - self.dx_dual)) / self.dx_dual
            if not cell_averages:
                self.A = A
            if np.max(np.abs(A - self.A)) > 1e-3:
                plt.figure()
                plt.plot(self.xc, self.A, label = 'cell average')
                plt.plot(self.xc, A, label = 'exact')
                plt.legend()
                plt.show()
                self.A = A

            self.diffusive_calc_dual_area()
            self.u = self.q / self.A_dual
            self.p = self.pe + self.K *( np.sqrt(self.A)/np.sqrt(self.A0) -1)
            self.forcing_term_steady()
            self.calc_area = self.calc_area_artery
            self.calc_der_area = self.calc_der_area_artery
            
        elif BC_type == "test_unsteady":
            self.q = (-np.sin(2*PI*self.xc) + np.sin(2*PI*np.roll(self.xc, -1)) )/((2*PI)**2)/self.dx_dual
            dA_dt =(np.cos(2*PI*np.roll(self.xd, 1))  - np.cos(2*PI*self.xd))/self.dx_primal
            q = np.cos(2*PI*self.xd) / (2*PI)
            if not cell_averages:
                self.q = q
            if (np.max(np.abs(self.q - q)) > 1e-3):
                plt.figure()
                plt.plot(self.xd, self.q, label = 'cell average')
                plt.plot(self.xd, q, label = 'exact')
                plt.legend()
                plt.show()
                self.q = q
            self.A = np.ones_like(self.A) * 4
            self.diffusive_calc_dual_area()
            self.u = self.q / self.A_dual
            self.forcing_term_conv = - np.sin(2*PI*self.xd)* np.cos(2*PI*self.xd) / (PI * 4)
            self.update_forcing = self.forcing_term_unsteady
            self.p = (self.pe + self.K * ((self.A/A0)**self.m_param-(self.A/A0)**self.n_param) +
                      self.Tau / (A0 * np.sqrt(self.A)) * dA_dt)
            if Tau != 0.0 or n_param != 0 or m_param != 0.5:
                self.calc_area = self.calc_area_general
                self.calc_der_area = self.calc_der_area_general
                self.artery_Tau0 = False
            else:
                self.calc_area = self.calc_area_artery
                self.calc_der_area = self.calc_der_area_artery

        if BC_type != "test_1":
            self.M_pv = self.calc_area(self.pv) * self.dx_primal
            self.dM_dpv = self.calc_der_area(self.pv) * self.dx_primal

        self.explicit = False
        self.diffusive_update_q = self.implicit_diffusive_update_q
        if False:
            print("Using explicit diffusion update")
            self.explicit = True
            self.diffusive_update_q  = self.explicit_diffusive_update_q


        self.q_starstar = np.copy(self.q)
        self.q_star = np.copy(self.q)

        self.flux_type = flux_type



    ####### CONVECTIVE STAGE #######

    ## The first stage of the algorithm involves the spatial discretization of the equation employing an explicit finite volume method
    # based on the Ducros or Kolgan-type numerical flux function


    # Minmod limiter to approximate Kolgan fluxes
    @staticmethod
    def minmod(a, b):
        return np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


    def convective_compute_slopes(self, q):
        # Equation 21: Reconstructed high order slopes for Kolgan method
        dq_minus = q - np.roll(q, 1)
        dq_plus = np.roll(q, -1) - q

        return self.minmod(dq_minus, dq_plus)


    def convective_kolgan_reconstruct(self, h):
        # Equation 20: Reconstructed interpolations of fluxes for Kolgan method
        # h can be q or u
        dh_minus = self.convective_compute_slopes(np.roll(h,1))
        dh_plus = self.convective_compute_slopes(h)

        h_R = np.roll(h, 1) + (self.xc - self.xd - self.dx_dual ) * dh_minus
        h_L = h + (self.xc- self.xd) * dh_plus
        return h_R, h_L

    def convective_compute_flux_kolgan(self, q, u):
        # Equation 22: Compute fluxes with Kolgan method
        q_R, q_L = self.convective_kolgan_reconstruct(q)
        u_R, u_L = self.convective_kolgan_reconstruct(u)
        a = 2 * np.maximum(np.abs(u_L), np.abs(u_R))
        return 0.25 * (u + np.roll(u, 1)) * \
            (q + np.roll(q, 1))- 0.5 * a * (q_L - q_R)

    def convective_compute_flux_ducros(self, q, u):
        # Equation 18: Compute fluxes for Ducros
        a = 2 * np.maximum(np.abs(u), np.abs(np.roll(u, 1)))
        return 0.25 * (u + np.roll(u, 1)) * (q + np.roll(q, 1)) - 0.5 * a * (q - np.roll(q, 1))

    def convective_update_q(self):
        # Equation 16: Update q_star at certain time step, solution of the convective stage
        self.q_star = np.copy(self.q)

        f_ip1 = self.flux_func(q=np.roll(self.q, -1), u=np.roll(self.u, -1))
        f_i = self.flux_func(q=self.q, u=self.u)

        self.q_star = self.q_star - self.dt / self.dx_primal * (f_ip1 - f_i) + self.forcing_term_conv * self.dt



    ####### DIFFUSIVE STAGE #######

    ## Once the first intermediate solution of the conservative variable 𝑞∗ is obtained within the convective stage, we can solve the diffusive stage following an implicit finite volume approach.
    # The discretization of the diffusive equation based on the backward in time centered in space scheme (BTCS)

    def diffusive_calc_dual_area(self):
        # The are at time step n has been estimated from the pressure at time step n-1
        # Arithmetic mean of neighbouring primal cells to obtain area in dual cells
        self.A_dual = (self.A + np.roll(self.A, -1)) / 2

    def calc_phi(self, A_value):
        # Equation 4: Viscoelastic behaviour of the wessel wall
        return self.Tau / (self.A0 * np.sqrt(A_value))

    def explicit_diffusive_update_q(self):
        # Equation 27: Solve matricial diffusive problem for explcit scenario
        
        N = self.n

        # Preallocate tridiagonal arrays
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)

        # Dual areas
        A_minus = np.roll(self.A_dual, 1)
        A_i = self.A_dual
        A_plus = np.roll(self.A_dual, -1)

        # dx
        dxa = self.dx_primal
        dxc = self.dx_dual
        dxb = np.roll(self.dx_primal, -1)

        # Phi values
        phi_m = self.calc_phi(A_minus)
        phi_i = self.calc_phi(A_i)
        phi_p = self.calc_phi(A_plus)

        phi_minus = 0.5 * (phi_i + phi_m)  # centered at i-1/2
        phi_plus = 0.5 * (phi_i + phi_p)

        q_i = self.q_star
        q_plus = np.roll(self.q_star, -1)
        q_minus = np.roll(self.q_star, 1)
        
        # Equation 23: Update q_starstar, solution of the diffusive stage
        q_starstar = (q_i + self.dt / self.rho * A_i / dxc * (phi_plus / dxb * (q_plus - q_i) -
                     phi_minus / dxa * (q_i - q_minus)) + self.dt * self.forcing_term_diff)
        self.q_starstar = q_starstar


    def implicit_diffusive_update_q(self):
        # Equation 27: Solve matricial diffusive problem for implicit scenario
        N = self.n

        # Preallocate tridiagonal arrays
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)

        # Dual areas
        A_minus = np.roll(self.A_dual, 1)
        A_i = self.A_dual
        A_plus = np.roll(self.A_dual, -1)

        # dx
        dxa = self.dx_primal
        dxc = self.dx_dual
        dxb = np.roll(self.dx_primal, -1)

        # Phi values
        phi_m = self.calc_phi(A_minus)
        phi_i = self.calc_phi(A_i)
        phi_p = self.calc_phi(A_plus)

        phi_minus = 0.5 * (phi_i + phi_m)  # centered at i-1/2
        phi_plus = 0.5 * (phi_i + phi_p)  # centered at i+1/2

        if self.nu != 0.0:
            print("we need to figure gamma out")
            a = -self.dt / self.rho * phi_minus / dxa
            b = (dxc / A_i+ (1 - self.beta) * self.dt * self.gamma * dxc / A_i +
                          self.dt / self.rho * (phi_plus / dxb + phi_minus / dxa))
            c= - self.dt / self.rho * phi_plus / dxb
        else:
            a = - self.theta * self.dt / self.rho * phi_minus / dxa
            c = - self.theta * self.dt / self.rho * phi_plus / dxb
            b = dxc / A_i - (a + c)

        # Sparse matrix and RHS
        diagonals = [
            a[1:],  # Lower diagonal (i-1)
            b,  # Main diagonal (i)
            c[:-1],  # Upper diagonal (i+1)
        ]
        offsets = [-1, 0, 1]
        D = diags(diagonals, offsets, shape=(N, N)).tolil()

        D[0, -1] = a[0]  # Periodic boundary condition
        D[-1, 0] = c[-1]
        q_i = self.q
        q_plus = np.roll(self.q, -1)
        q_minus = np.roll(self.q, 1)

        # Equation 28: Update q_starstar, solution of the diffusive stage
        rhs = ((self.q_star + self.dt * self.forcing_term_diff) * dxc / A_i  + (1 - self.theta) * self.dt / self.rho *
               (phi_plus / dxb * (q_plus - q_i) - phi_minus / dxa * (q_i - q_minus)))
        q_starstar = spsolve(D.tocsc(), rhs)

        # Restore boundaries
        self.q_starstar = q_starstar



    ####### PRESSURE STAGE #######

    def calc_area_artery(self, p):
        # Equation 40: Calculate area based on pressure for an elastic artery (derived from tube law with m = 0.5, n = 0)
        A = self.A0 * (1 + (p - self.pe) / self.K) ** 2
        return A

    def calc_der_area_artery(self, p):
        # Calculate the derivative of area with respect to pressure for an elastic artery
        dA_dp = (2 * self.A0 / self.K) * (1 + (p - self.pe) / self.K)
        return dA_dp

    # The 2 following functions will be used for the double newton iteration performed to solve the pressure stage
    def calc_area_general(self, p):
        # Calculate area based on pressure for a general case
        def tube_law_residual(A):
            elastic = self.K * (np.power((A / self.A0), self.m_param) - np.power(A/ self.A0, self.n_param))
            visco = -self.Tau / (np.sqrt(A)* self.A0 *self.dt) * (self.A - A)
            return self.pe + elastic + visco - p
        # Use a root-finding method to find the area A that satisfies the tube law
        A = root(tube_law_residual, self.A, method='hybr', options={'xtol': 1e-8, 'maxfev': 10000})
        if  not A.success:
            print(p)
            print("Root finding did not converge for area calculation.")
            print("Residual:", A.fun)
            print("Message:", A.message)
            raise RuntimeError("Area calculation failed to converge.")
        return np.array(A.x)

    def calc_der_area_general(self, p):
        eps = 1e-6# Small perturbation for numerical differentiation
        A_p = self.calc_area_general(p + eps)
        A_m = self.calc_area_general(p - eps)
        dA_dp = (A_p - A_m) / (2 * eps)
        return dA_dp
    

    def pressure_stage_updates(self):
        
        # Solve pressure stage subsystem: equations 33-34

        # i goes for 1 to Np in paper 0 to Np -1 in python
        N = self.n

        # 1. UPDATE PRESSURE: Solve pressure problem
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        b_rhs = np.zeros(N)

        dxa = np.roll(self.dx_dual,1)
        dxb = self.dx_dual

        A_minus = np.roll(self.A_dual,1)
        A_i = self.A_dual

        self.p_tilde = self.pe + self.K * ((self.A / self.A0) ** self.m_param - 1)

        if self.nu != 0.0:
            print("we need to figure gamma out")
            denom_plus = self.rho * (1 + self.beta * self.dt * self.gamma)
            denom_minus = self.rho * (1 + self.beta * self.dt * np.roll(self.gamma,1))
        else:
            denom_plus = self.rho * np.ones(N)
            denom_minus = self.rho * np.ones(N)

        a = -(self.dt) ** 2 / (dxa * denom_minus) * A_minus
        c = -(self.dt) ** 2 / (dxb * denom_plus) * A_i
        b = - (a + c)

        diagonals = [
            a[1:],  # Lower diagonal (i-1)
            b,  # Main diagonal (i)
            c[:-1],  # Upper diagonal (i+1)
        ]
        offsets = [-1, 0, 1]
        T = diags(diagonals, offsets, shape=(N, N)).tolil()

        T[0, -1] = a[0]  # Periodic boundary condition
        T[-1, 0] = c[-1]

        M = self.A * self.dx_primal
        T = T.tocsr()
        p = self.p
        # 1.B Compute RHS vector b
        b_rhs = M - self.theta * self.dt * self.rho * (
                (self.q_starstar + self.dt * self.forcing_term_press) / denom_plus - \
                np.roll(self.q_starstar + self.dt* self.forcing_term_press,1) / denom_minus) \
            - self.theta * self.dt ** 2 * ((1 / dxb) * (A_i / denom_plus) * ( np.roll(self.p - self.p_tilde,-1) - (self.p - self.p_tilde))
            - (1 / dxa) * (A_minus / denom_minus) * ((self.p - self.p_tilde) - np.roll(self.p - self.p_tilde, 1))) \
            - (1 - self.theta) * self.dt * (self.q - np.roll(self.q,1)) \
            - (1 - self.theta) * self.theta * T @ self.p

        T *= self.theta**2
        

        # Nested Newton methodology to solve pressure subsystem to account for non linearities
        # Split the diagonal nonlinear terms and then linearize the nonlinear contributions in sequence to derive a nested iterative method. 
        # Assuming the first derivative of 𝐌(p_n+1) to be a function of bounded variations, we can use the Jordan decomposition and express 
        # the first derivative of 𝐌(pn+1) as the diﬀerence of two nonnegative, nondecreasing bounded functions.
        # Note that although this has been implemented, the non-linearity should only affect physical scenarios involving elastic veins, not arteries
        
        if self.artery_Tau0:
            conv = False
            # If the artery is in the initial state, we can use the artery law
            for i in range(self.n):
                res = M = self.calc_area_artery(p) * self.dx_primal
                if np.linalg.norm(res) <= 1e-14:
                    conv = True
                    break
                dM_dp = self.calc_der_area_artery(p) * self.dx_primal
                res = M + T @ p - b_rhs
                J = diags(dM_dp) + T
                dp = spsolve(J, -res)
                p += dp
        else:
            def get_M1(p):
                M = self.calc_area(p) * self.dx_primal  # total M(p)
                M1 = np.where(
                    p <= self.pv,
                    M,
                    self.M_pv + self.dM_dpv * (p - self.pv)
                )
                return M1, M

            def get_M1_and_M2(p):
                M1, M = get_M1(p)
                M2 = M1 - M
                return M1, M2

            def get_R(p):
                dM_dp = self.calc_der_area(p) * self.dx_primal
                dM1_dp = np.where(
                    p <= self.pv,
                    dM_dp,
                    self.dM_dpv
                )
                return dM1_dp, dM_dp

            def get_RQ(p):
                dM1_dp, dM_dp = get_R(p)
                return dM1_dp, dM1_dp - dM_dp


            # Clip initial guess: p0 <= pv
            p = np.zeros_like(p)
            conv = False

            for n in range(self.n):  # Outer loop (θ₂ linearized)
                M1, M2 = get_M1_and_M2(p)
                dM1_dp, dM2_dp = get_RQ(p)

                R = diags(dM1_dp)
                Q = diags(dM2_dp)
                J = R + T - Q

                res_n = M1 - M2 + T @ p - b_rhs
                if np.linalg.norm(res_n) <= 1e-8:
                    conv = True
                    break

                d = b_rhs + M2 - Q @ p

                for m in range(self.n):  # Inner loop (θ₁ linearized)
                    res_nm = M1 + (T - Q) @ p - d
                    #print(f"Outer {n}, Inner {m}, Residual: {np.linalg.norm(res_nm)}")
                    if np.linalg.norm(res_nm) <= 1e-10:
                        break

                    f = R @ p - M1 + d
                    try:
                        p_new = spsolve(J, f)
                    except Exception as e:
                        raise RuntimeError("Linear solve failed") from e

                    if np.isnan(p_new).any():
                        print("NaN encountered during inner iteration")
                        print("Jacobian:\n", J.toarray())
                        raise RuntimeError("NaN in pressure stage")

                    p = p_new
                    M1, _ = get_M1(p)
                    dM1_dp, _ = get_R(p)
                    R = diags(dM1_dp)
                    J = R + T - Q

            if not conv:
                print("Convergence not reached in pressure stage with tube law")
                print("Residual norm:", np.linalg.norm(res_n))

        return p
    
    

    ##### CORRECTION STAGE #####

    def correction_stage(self, p_updated):

        # 1. Convective, diffusive and pressure stages have been completed thus far

        N = self.n

        A_i = self.A_dual
        dx = self.dx_dual


        # 2. UPDATE Q: Equation 31

        if self.nu != 0.0:
            print("We need to figure gamma out")
            q_new = (1 / (1 +self.beta * self.dt * self.gamma)) * \
                          (self.q_starstar + self.dt * self.forcing_term_press \
                           - self.theta * ((self.dt * A_i) / (self.rho * dx)) * (np.roll(p_updated,-1) - p_updated) \
                           + ((self.dt * A_i) / (self.rho * dx)) * (np.roll(self.p - self.p_tilde, -1) - (self.p - self.p_tilde))) \
                           - (1 - self.theta) * ((self.dt * A_i) / (self.rho * dx)) * (np.roll(self.p, -1) - self.p)
        else:
            q_new = (self.q_starstar + self.dt * self.forcing_term_press\
                           - self.theta * ((self.dt * A_i) / (self.rho * dx)) * (np.roll(p_updated,-1) - p_updated) \
                           + ((self.dt * A_i) / (self.rho * dx)) * (np.roll(self.p - self.p_tilde, -1) - (self.p - self.p_tilde))) \
                           - (1 - self.theta) * ((self.dt * A_i) / (self.rho * dx)) * (np.roll(self.p, -1) - self.p)



        # 3. UPDATE AREA: Equation 40
        # In case an elastic artery is considered, i.e. for 𝑚 = 0.5, 𝑛 = 0 and Γ = 0
        self.q = q_new
        self.q_star = np.copy(self.q)
        self.q_starstar = np.copy(self.q)
        self.A = self.calc_area(p_updated)
        self.p = p_updated
        self.diffusive_calc_dual_area()
        self.compute_gamma()
        self.u = self.q / self.A_dual

    def update_forcing_zeros(self):
        pass

    def forcing_term_steady(self):
        # Verify that expected forcing term on paper matches with observed  forcing term (numerical) for steady test case

        self.forcing_term_conv =  (np.sin( 2 * PI * np.roll(self.xc,-1)) - np.sin( 2 * PI * self.xc))/self.dx_dual
        integral_press = self.K/(3*np.sqrt(self.A0)* self.rho*(np.sin(2 * PI * self.xc) + 4)**(1.5))
        self.forcing_term_press =  (-integral_press+np.roll(integral_press, - 1)) / self.dx_dual


        forcing_term_conv = 2 * PI * np.cos(2 * PI * self.xd)
        forcing_term_press = (- PI * self.K * np.cos(2 * PI * self.xd) /
                                  (np.sqrt(self.A0) * self.rho * ((np.sin(2 * PI * self.xd) + 4)**2.5)))

        if ((self.forcing_term_conv - forcing_term_conv).max() > 1e-2 or
            (self.forcing_term_press - forcing_term_press).max() > 1e-2) and self.cell_averages:
            print("Forcing terms do not match expected values. Check implementation.")
            print(np.max(np.abs(self.forcing_term_conv - forcing_term_conv)))
            print(np.max(np.abs(self.forcing_term_press - forcing_term_press)))

        if not self.cell_averages:
            self.forcing_term_conv = forcing_term_conv
            self.forcing_term_press = forcing_term_press
        """
        plt.figure()
        plt.title("forcing term conv")
        plt.plot(self.xd, forcing_term_conv, label = "instant")
        plt.plot(self.xd, self.forcing_term_conv, label ="average cell", linestyle = "--" )
        plt.legend()
        plt.show()
        plt.figure()
        plt.title("forcing term press")
        plt.plot(self.xd, forcing_term_press,label = "instant")
        plt.plot(self.xd, self.forcing_term_press,label = "average cell")
        plt.legend()
        plt.show()
        """

    def forcing_term_unsteady(self):
        # Verify that expected forcing term on paper matches with observed  forcing term (numerical) forun steady test case

        t = self.t
        A = t * np.sin(2 * PI * self.xc) + 4
        A_dual = t * np.sin(2 * PI * self.xd) + 4

        conv_integral= (np.power(t * np.cos(2 * PI * self.xc), 2) - 2 * 4 * A) / ((2 * PI * t)**2 * A)
        self.forcing_term_conv = -(np.roll(conv_integral, -1) - conv_integral) / self.dx_dual
        dif_integral = self.Tau * np.sqrt(A)*(A + 3 * 4)/ (3 * self.A0 * self.rho * t)
        self.forcing_term_diff = (np.roll(dif_integral, -1) - dif_integral) / self.dx_dual
        # self.forcing_term_diff = 0
        press_integral = self.K * (A)**(3/2)/(3 * np.sqrt(self.A0) * self.rho )
        self.forcing_term_press = -(press_integral - np.roll(press_integral, -1)) / self.dx_dual

        forcing_term_conv =- (- t * np.cos(2 * PI * self.xd) ** 3 / (2 * PI * A_dual ** 2) - \
                                 np.sin(2 * PI * self.xd) * np.cos(2 * PI * self.xd) / (PI * A_dual))

        self.forcing_term_diff = (PI * self.Tau * (A_dual + 4) * np.cos(2 * PI * self.xd) /
                              (self.A0 * self.rho * np.sqrt(A_dual)))

        forcing_term_press = PI * self.K * t * np.sqrt(A_dual) * np.cos(2 * PI * self.xd) / (np.sqrt(self.A0) * self.rho)

        if ((self.forcing_term_conv - forcing_term_conv).max() > 1e-2 or
           (self.forcing_term_press - forcing_term_press).max() > 1e-2) and self.cell_averages:
            print("Forcing terms do not match expected values. Check implementation.")
            print(np.max(np.abs(self.forcing_term_conv - forcing_term_conv)))
            print(np.max(np.abs(self.forcing_term_press - forcing_term_press)))

        if not self.cell_averages:
            self.forcing_term_conv = forcing_term_conv
            self.forcing_term_press = forcing_term_press

    def compute_gamma(self):
        # Compute gamma based on the current state of the system: we are not using this really as we only consider 0 viscosity scenarios thus the source term s disappeas
        self.gamma = self.s / self.A_dual

    def update_CFL(self):
        # Update CFL condition and time step if CFL condition not met
        dx = self.dx_dual
        mask = self.u != 0

        # Select only corresponding dx and u values where u != 0
        dx_nonzero = dx[mask]
        u_nonzero = self.u[mask]
        CFL_cond = np.inf  # Initialize CFL condition

        if u_nonzero.size > 0:
            CFL_cond = 0.5 * np.min(dx_nonzero / np.abs(u_nonzero))
            self.dt = min(self.dt_initial, CFL_cond)

        if self.explicit:
            phi = self.calc_phi(self.A_dual)
            CFL_cond = 0.5 * np.min(self.rho * np.power(self.dx_dual, 2)/(self.A_dual * phi))
            self.dt = min(self.dt, CFL_cond)
            

    def test_convection(self):
        # VALIDATION TEST 1: Purely convective
        
        print( "dt = ", self.dt, "tF = ", self.tF)
        q_solutions = []
        q_star_solutions = []

        q_star_solutions.append(self.q_star)
        i = 0
        t = 0
        t_sol = [0]
        while t < self.tF:
            self.update_CFL()

            self.convective_update_q()

            q_star_solutions.append(self.q_star)
            t += self.dt
            t_sol.append(t)
            self.q = np.copy(self.q_star) # Use q_star solution at time step n as "total" q solution
            i += 1
            self.u = self.q / self.A_dual


            # u_solutions[t_step, :] = q_new/A_new
        return q_star_solutions, t_sol
    
    def test_diffusion(self):
        # VALIDATION TEST 2: Purely diffusive
        
        print( "dt = ", self.dt, "tF = ", self.tF)
        q_solutions = []
        q_star_star_solutions = []
        q_star_star_solutions.append(self.q_starstar)
        i = 0
        t = 0
        t_sol = [0]
        while t < self.tF:

            # self.convective_update_q()
            self.diffusive_update_q()

            # q_star_solutions.append(self.q_star)
            q_star_star_solutions.append(self.q_starstar)
            t += self.dt
            t_sol.append(t)
            self.q_star = np.copy(self.q_starstar)
            self.q = np.copy(self.q_starstar) # Use q_star_star solution at time step n as "total" q solution
            i += 1
            self.u = self.q / self.A_dual


            # u_solutions[t_step, :] = q_new/A_new
        return q_star_star_solutions, t_sol

    def solve_blood_vessel(self):
        
        # TESTS 3 and 4 + GENERAL SOLVING PROCEDURE
        
        n_steps = int(self.tF / self.dt)

        q_solutions =[self.q]
        u_solutions = [self.u]
        p_solutions = [self.p]
        A_solutions = [self.A]
        q_starstar_solutions = [self.q_starstar]
        q_star_solutions = [self.q_star]
        i = 0
        t_sol = [0]

        while self.t < self.tF:
            self.update_CFL()
            self.convective_update_q()
            self.diffusive_update_q()
            p_new = self.pressure_stage_updates()
            self.correction_stage(p_new)

            q_solutions.append(self.q)
            A_solutions.append(self.A)
            p_solutions.append(p_new)
            q_starstar_solutions.append(self.q_starstar)
            q_star_solutions.append(self.q_star)
            u_solutions.append(self.q / self.A_dual)
            print(self.t, " / ", self.tF, " dt: ", self.dt, " q_max: ", np.max(self.q), " A_max: ", np.max(self.A), " p_max: ", np.max(p_new))
            self.t += self.dt
            t_sol.append(self.t)
            self.update_forcing()


        return q_solutions, A_solutions, p_solutions, q_star_solutions, q_starstar_solutions, u_solutions, t_sol


if __name__ == "__main__":
    L = 1.0
    A0ref = 3.1416e-4
    Kref = 80
    pe = 0
    rho = 1050
    tF = 0.1
    dt = 1e-3

    grid = Grid1D(L=1.0, Np=30, stretch_factor=1, stretch_type='uniform')

    blood_solution = BloodVessel1Dperiodic(grid=grid, flux_type='kolgan', tF=0.1, dt=dt, rho=rho, Tau=1.0, A0=A0ref,
                                           K=Kref, BC_type="test_unsteady", beta=0.0, nu = 0.0, xi = 0.0, theta = 0.5)
    q_solutions, A_solutions, p_solutions, q_star_solutions, q_starstar_solutions, u_solutions, t_sol = blood_solution.solve_blood_vessel()

    q_final = q_solutions[0]
    A_final = A_solutions[0]
    p_final = p_solutions[0]

    # Get spatial positions
    x_dual = blood_solution.xd  # for q (dual centers)
    x_primal = blood_solution.xc  # for A and p (primal centers)

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot q vs dual centers
    axs[0].plot(x_dual, q_final, label='Flow rate (q)', color='blue')
    axs[0].set_ylabel('q [m³/s]')
    axs[0].legend()
    axs[0].grid(True)

    # Plot A vs primal centers
    axs[1].plot(x_primal, A_final, label='Cross-sectional Area (A)', color='green')
    axs[1].set_ylabel('A/A0')
    axs[1].legend()
    axs[1].grid(True)

    # Plot p vs primal centers
    axs[2].plot(x_primal, p_final, label='Pressure (p)', color='red')
    axs[2].set_xlabel('Position [m]')
    axs[2].set_ylabel('p [Pa]')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    q_final = q_solutions[-1]
    A_final = A_solutions[-1]
    p_final = p_solutions[-1]

    # Get spatial positions
    x_dual = blood_solution.xd  # for q (dual centers)
    x_primal = blood_solution.xc  # for A and p (primal centers)

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot q vs dual centers
    axs[0].plot(x_dual, q_final, label='Flow rate (q)', color='blue')
    axs[0].set_ylabel('q [m³/s]')
    axs[0].legend()
    axs[0].grid(True)

    # Plot A vs primal centers
    axs[1].plot(x_primal, A_final, label='Cross-sectional Area (A)', color='green')
    axs[1].set_ylabel('A/A0')
    axs[1].legend()
    axs[1].grid(True)

    # Plot p vs primal centers
    axs[2].plot(x_primal, p_final, label='Pressure (p)', color='red')
    axs[2].set_xlabel('Position [m]')
    axs[2].set_ylabel('p [Pa]')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

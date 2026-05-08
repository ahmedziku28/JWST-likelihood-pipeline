# growth_factor.py
#
# Solves the linear matter perturbation equation, converted to ln(a).
#
# Starting equation (cosmic time form, Mo van den Bosch & White 2010 eq 4.46):
#
#   delta_ddot + 2H*delta_dot = 4*pi*G * rho_m * delta
#
# After substituting d/dt = H * d/d(ln a)  [Linder 2005, PhysRevD 72, 043529]:
#
#   delta'' + (2 + H'/H) * delta' = (3/2) * H0^2 * Omega_m / (a^3 * H^2) * delta
#
# where primes denote d/d(ln a).

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

    
    
def compute_growth_factor(cosmo, z_array):
    """
    Compute D(z) = delta(z) / delta(z=0), normalized so D(z=0) = 1.

    Parameters
    ----------
    cosmo   : classy.Class, already initialized and computed
    z_array : array_like, redshifts at which you want D(z)

    Returns
    -------
    D : np.ndarray, same length as z_array
    """

    
    # ------------------------------------------------------------------
    # Pull fixed numbers from CLASS
    # ------------------------------------------------------------------

    # H0 in CLASS internal units: 1/Mpc
    H0_class = cosmo.Hubble(0)

    # Total matter fraction
    Omega_m_total = cosmo.Omega_m()

    # ------------------------------------------------------------------
    # Translator: ln(a) → H(z) via CLASS
    # ------------------------------------------------------------------
    # The ODE runs in ln(a). CLASS speaks z.
    # We convert at every step the integrator visits.

    def H_from_lna(lna):
        a = np.exp(lna)
        z = 1.0 / a - 1.0
        z = max(z, 0.0)        # guard against tiny overshoot
        return cosmo.Hubble(z)

    # ------------------------------------------------------------------
    # Numerical derivative dH/d(ln a) via centered finite difference
    # ------------------------------------------------------------------
    # dH/d(ln a) ≈ [H(ln a + ε) - H(ln a - ε)] / (2ε)
    # Centered difference is second-order accurate: error ~ ε²
    # ε = 1e-4 gives errors ~ 1e-8, completely negligible

    def dH_dlna(lna, epsilon=1e-4):
        H_forward  = H_from_lna(lna + epsilon)
        H_backward = H_from_lna(lna - epsilon)
        derivative = (H_forward - H_backward) / (2.0 * epsilon)
        return derivative

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    # Second-order ODE rewritten as two coupled first-order equations.
    #
    # state[0] = overdensity             = delta
    # state[1] = overdensity_growth_rate = d(delta)/d(ln a)
    #
    # Their rates of change (both are d/d(ln a)):
    #   d(overdensity)/d(ln a)             = overdensity_growth_rate
    #   d(overdensity_growth_rate)/d(ln a) = gravity*overdensity
    #                                       - friction*overdensity_growth_rate

    def growth_ode(lna, state):

        overdensity             = state[0]
        overdensity_growth_rate = state[1]

        H = H_from_lna(lna)
        a = np.exp(lna)

        # Friction: how hard expansion resists gravitational collapse.
        # "2.0" always comes from the ln(a) coordinate transformation.
        # dH_dlna/H tells us how fast H is changing — faster fall = less friction.
        friction_coefficient = 2.0 + dH_dlna(lna) / H

        # Gravitational source: 4*pi*G*rho_m rewritten using
        # rho_m = (3 H0^2 Omega_m / 8*pi*G) * a^{-3}
        gravitational_source = (1.5 * Omega_m_total * H0_class**2
                                / (a**3 * H**2))

        # Equation 1: trivially, rate of change of overdensity = growth rate
        d_overdensity = overdensity_growth_rate

        # Equation 2: the actual physics — gravity in, expansion out
        d_overdensity_growth_rate = (gravitational_source * overdensity
                                    - friction_coefficient * overdensity_growth_rate)

        return [d_overdensity, d_overdensity_growth_rate]

    # ------------------------------------------------------------------
    # Initial conditions at z = 3200 (The dawn of Matter Domination)
    # ------------------------------------------------------------------
    # Analytic matter domination solution: delta ∝ a
    # In ln(a) variables: d(delta)/d(ln a) = d(a)/d(ln a) = a
    # So both state variables equal a_ini at the start.
    # Absolute amplitude doesn't matter — we normalize D(z=0)=1 at the end.

    z_start = 3200.0
    a_start = 1.0 / (1.0 + z_start)  # a ≈ 0.0003124
    lna_start = np.log(a_start)      # ln(a) ≈ -8.07

    lna_today = 0.0                  # a = 1, z = 0

    # If delta ∝ a, then d(delta)/d(ln a) = a
    # This seeds the 'Growing Mode' precisely as the universe 
    # exits the Radiation Era and begins clumping matter.
    initial_state = [a_start, a_start]

    # ------------------------------------------------------------------
    # Build the evaluation grid
    # ------------------------------------------------------------------
    z_arr   = np.atleast_1d(np.asarray(z_array, dtype=float))
    lna_req = np.log(1.0 / (1.0 + z_arr))

    # Always include a dense internal grid of at least 500 points.
    # This guarantees the interpolator always has plenty of points
    # regardless of how many redshifts the user requests.
    # Without this, requesting a single redshift gives only 2 points
    # and the cubic spline crashes.
    lna_dense_grid = np.linspace(lna_start, lna_today, 1000)
    
    lna_merged_grid = np.union1d(lna_dense_grid, lna_req)

    # Merge dense grid + requested points + endpoints, sort ascending
    lna_output_grid = np.sort(
        lna_merged_grid
    )

    # ------------------------------------------------------------------
    # Integrate the ODE
    # ------------------------------------------------------------------
    # DOP853: 8th-order Runge-Kutta, adaptive step size.
    # Tight tolerances: D(z) accurate to better than 0.01%.

    solution = solve_ivp(
        growth_ode,
        t_span = [lna_start, lna_today],
        y0     = initial_state,
        method = 'DOP853',
        t_eval = lna_output_grid,
        rtol   = 1e-10,
        atol   = 1e-12
    )

    if not solution.success:
        raise RuntimeError(
            f"Growth factor ODE failed: {solution.message}"
        )

    # ------------------------------------------------------------------
    # Normalize so D(z=0) = 1 by construction
    # ------------------------------------------------------------------
    # The ODE solution delta(z) has an arbitrary amplitude set by our
    # initial condition choice. Dividing by delta(z=0) removes that
    # arbitrariness and enforces the universal convention D(0) = 1.

    overdensity_solution = solution.y[0]
    lna_solution         = solution.t

    index_today       = np.argmin(np.abs(lna_solution - 0.0))
    overdensity_today = overdensity_solution[index_today]

    D_normalized = overdensity_solution / overdensity_today

    # ------------------------------------------------------------------
    # Interpolate to the exactly requested redshift values
    # ------------------------------------------------------------------
    D_interpolator = interp1d(
        lna_solution,
        D_normalized,
        kind         = 'cubic',
        bounds_error = True    # crash loudly if asked to extrapolate
    )

    return D_interpolator(lna_req)
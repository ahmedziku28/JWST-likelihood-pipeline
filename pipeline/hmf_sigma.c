/* pipeline/hmf_sigma.c
 *
 * Computes sigma^2(M) and dsigma^2/dR for an array of Lagrangian radii,
 * given P(k) on a log-spaced k grid.
 *
 * Compile:
 *   gcc -O3 -march=native -ffast-math -shared -fPIC \
 *       -o hmf_sigma.so hmf_sigma.c -lm
 *
 * Called from Python via ctypes (see hmf_plugin.py).
 *
 * All units: k [Mpc^-1], P [Mpc^3], R [Mpc].
 * sigma^2 and dsigma^2/dR are dimensionless and [Mpc^-1] respectively.
 */

#include <math.h>
#include <stdlib.h>

/* After the #include lines, add: */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Top-hat window W(x) and its derivative dW/dx.
 * Taylor expansion for x < 1e-4 to avoid catastrophic cancellation. */
static inline double tophat_W(double x) {
    if (x > 8e-5)
        return 3.0*(sin(x) - x*cos(x)) / (x*x*x);
    else
        return (1.0 - ((x*x)/10.0));
}

static inline double tophat_dWdx(double x) {
    if (x > 8e-5)
        return (3.0*x*x*sin(x) - 9.0*sin(x) + 9.0*x*cos(x)) / (x*x*x*x);
    else
        return ((-x / 5.0) + ((x*x*x)/(70.0)));
}

/*
 * compute_sigma_batch
 *
 * For each radius R_grid[j], integrates:
 *
 *   sigma2[j]     = (1/2pi^2) * integral k^3 * P(k) * W^2(kR) d(lnk)
 *   dsigma2_dR[j] = (1/2pi^2) * integral k^4 * P(k) * W(kR) * dW/d(kR) d(lnk)
 *
 * using the trapezoid rule on the provided log-spaced k grid.
 *
 * Parameters:
 *   k_grid, lnk_grid, Pk  : arrays of length n_k
 *   R_grid                 : array of length n_M
 *   sigma2, dsigma2_dR     : output arrays of length n_M (caller allocates)
 *   n_k, n_M               : grid sizes
 */
void compute_sigma_batch(
    const double* k_grid,
    const double* lnk_grid,
    const double* Pk,
    const double* R_grid,
    double*       sigma2,
    double*       dsigma2_dR,
    int           n_k,
    int           n_M
) {
    const double inv_2pi2 = 1.0 / (2.0 * M_PI * M_PI);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n_M; j++) {
        double R   = R_grid[j];
        double s   = 0.0;
        double ds  = 0.0;

        /* First point */
        double kR   = k_grid[0] * R;
        double W    = tophat_W(kR);
        double dW   = tophat_dWdx(kR);
        double k3   = k_grid[0]*k_grid[0]*k_grid[0];
        double k4   = k3 * k_grid[0];
        double fs_p = k3 * Pk[0] * W * W;
        double fd_p = k4 * Pk[0] * W * dW;

        for (int i = 1; i < n_k; i++) {
            kR      = k_grid[i] * R;
            W       = tophat_W(kR);
            dW      = tophat_dWdx(kR);
            k3      = k_grid[i]*k_grid[i]*k_grid[i];
            k4      = k3 * k_grid[i];
            double fs = k3 * Pk[i] * W * W;
            double fd = k4 * Pk[i] * W * dW;
            double dlnk = lnk_grid[i] - lnk_grid[i-1];

            s  += 0.5 * (fs_p + fs) * dlnk;
            ds += 0.5 * (fd_p + fd) * dlnk;

            fs_p = fs;
            fd_p = fd;
        }

        sigma2[j]     = s  * inv_2pi2;
        dsigma2_dR[j] = ds * 2.0 * inv_2pi2;
    }
}
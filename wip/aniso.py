import numpy as np
from scipy.stats import ncx2, norm
from numpy.polynomial.legendre import leggauss


def h_aniso_rz(rho0, z0, R, sigma_rho, sigma_z, n_gl=32):
    """
    Vectorized paired evaluation of the convolution 3D unit sphere (with radius R)
    with anisotropic Gaussian kernel in the transverse (rho) and axial (z) directions.

    Parameters
    ----------
    rho0 : array_like, shape (n,)
        Transverse radii (sqrt(x^2+y^2)) for n query points.
    z0   : array_like, shape (n,)
        Axial coordinates for the same n query points.
    R : float
        Sphere radius.
    sigma_rho : float
        Gaussian std in x,y.
    sigma_z   : float
        Gaussian std in z.
    n_gl : int, optional
        Number of Gauss–Legendre nodes (default 96). 48–128 is usually plenty.

    Returns
    -------
    h : ndarray, shape (n,)
        Convolved values at each paired (rho0[i], z0[i]).
    """
    rho0 = np.asarray(rho0, dtype=float).ravel()
    z0 = np.asarray(z0, dtype=float).ravel()
    if rho0.shape != z0.shape:
        raise ValueError("rho0 and z0 must have the same shape (paired inputs).")
    n = rho0.size

    # Gauss–Legendre nodes/weights on [-1, 1]
    t, w = leggauss(n_gl)  # (n_gl,), (n_gl,)
    z = R * t  # map to [-R, R]

    # Slice radius ρ(z') = sqrt(R^2 - z'^2) = R * sqrt(1 - t^2)
    rho_slice = R * np.sqrt(np.maximum(1.0 - t**2, 0.0))  # (n_gl,)

    # Arguments for the special functions (broadcasted over n × n_gl)
    x = (rho_slice / sigma_rho) ** 2  # (n_gl,)
    nc = (rho0 / sigma_rho) ** 2  # (n,)

    # In-disk probability in (x,y): noncentral chi-square CDF with df=2
    F_xy = ncx2.cdf(x[None, :], df=2, nc=nc[:, None])  # (n, n_gl)

    # 1D Gaussian along z' centered at each z0[i]
    phi = norm.pdf(z[None, :], loc=z0[:, None], scale=sigma_z)  # (n, n_gl)

    # Gauss–Legendre integral: sum w_j * f(t_j) * R
    # the extra R factor is because we are integrating from -R to R, not from -1 to 1
    h = (F_xy * phi * w[None, :]).sum(axis=1) * R  # (n,)
    return h


# %%

s_rho = 1.5
s_z = 4.0
Rt = 18.5
n = 100

n_gauss_leg = 32

# %%

N = 301
x, dx = np.linspace(-1.5 * Rt, 1.5 * Rt, N, retstep=True)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
RHO = np.sqrt(X**2 + Y**2)
R = np.sqrt(X**2 + Y**2 + Z**2)

img = np.zeros((N, N, N), dtype=float)
img[R <= Rt] = 1

from scipy.ndimage import gaussian_filter

img_sm = gaussian_filter(img, sigma=(s_rho / dx, s_rho / dx, s_z / dx))

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    4, 1, figsize=(8, 8), sharex=True, sharey=True, layout="constrained"
)

for i, iz in enumerate([N // 6, N // 3, N // 2]):
    ax[i].plot(x, img_sm[:, N // 2, iz])
    ax[i].plot(
        x,
        h_aniso_rz(
            np.abs(x), np.abs(np.full(x.shape, x[iz])), Rt, s_rho, s_z, n_gl=n_gauss_leg
        ),
        ls="--",
        label="h_aniso",
    )
    ax[i].set_xlabel("x")
    ax[i].set_title(f"radial profile, z = {x[iz]:.2f}", fontsize="medium")

ax[-1].plot(x, img_sm[N // 2, N // 2, :])
ax[-1].plot(
    x,
    h_aniso_rz(np.zeros_like(x), x, Rt, s_rho, s_z, n_gl=n_gauss_leg),
    ls="--",
    label="h_aniso",
)

ax[-1].set_xlabel("z")
ax[-1].set_title(f"axial profile, x = {x[N//2]:.2f}", fontsize="medium")

for axx in ax:
    axx.grid(ls=":")
fig.show()

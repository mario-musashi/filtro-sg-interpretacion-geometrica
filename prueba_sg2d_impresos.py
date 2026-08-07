#!/usr/bin/env python3
"""Prueba del pipeline SG 2D con defectos impresos (sintético + real placeholder).

Correcciones respecto a sg_2D.ipynb:
- Se resta la base (5 mm) antes de SG: el estimador asume fondo cero (A = -f0).
- sigma se estima con la fórmula de la tesis: sigma = sqrt(A/|kappa|),
  con kappa = autovalores del Hessiano H = [[2a, c],[c, 2b]].
  (El notebook original usaba autovalores de Q = H/2, lo que daba sesgo sqrt(2).)
"""
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

L = 150
STEP = 0.2
WINDOW_SIZE = 21
RUTA_PARAMS = f"/home/mario-ross/Proyectos/Tesis/python_pruebas/stl/parametros_{L}mm.csv"

gt = pd.read_csv(RUTA_PARAMS)


def sg2d_jacobian_projection_cv2(Z, window_size=11):
    assert window_size % 2 == 1, "La ventana debe ser impar"
    hw = window_size // 2
    y_idx, x_idx = np.mgrid[-hw:hw+1, -hw:hw+1]
    regressors = np.stack([
        x_idx**2, y_idx**2, x_idx*y_idx, x_idx, y_idx,
        np.ones_like(x_idx)
    ], axis=-1)
    A = regressors.reshape(-1, 6)
    pseudo = np.linalg.inv(A.T @ A) @ A.T
    kernels = pseudo.reshape(6, window_size, window_size)
    return np.stack([cv2.filter2D(Z, -1, k, borderType=cv2.BORDER_REFLECT)
                     for k in kernels], axis=-1)


def estimate_bump_parameters(centers, coeffs, step):
    """A, sigma1, sigma2 (mm) y theta. Sigma via autovalores del Hessiano."""
    results = []
    for (i0, j0) in centers:
        ii, jj = int(round(i0)), int(round(j0))
        a, b, c, d, e, f0 = coeffs[ii, jj]
        H = np.array([[2*a, c], [c, 2*b]])
        grad = np.array([d, e])
        dx, dy = -np.linalg.solve(H, grad)
        vi, vj = ii + dy, jj + dx
        xv, yv = vj - jj, vi - ii
        A_est = abs(-(a*xv**2 + b*yv**2 + c*xv*yv + d*xv + e*yv + f0))
        k1, k2 = np.linalg.eigvalsh(H)
        w1 = np.sqrt(A_est / max(abs(k1), 1e-12)) * step
        w2 = np.sqrt(A_est / max(abs(k2), 1e-12)) * step
        sigma1, sigma2 = max(w1, w2), min(w1, w2)
        results.append({'center': (vi, vj), 'A': A_est,
                        'sigma1': sigma1, 'sigma2': sigma2})
    return results


def add_rotated_gaussian(Z, x, y, xc, yc, sx, sy, A, th_deg):
    th = np.deg2rad(th_deg)
    X0, Y0 = x - xc, y - yc
    Xr =  np.cos(th) * X0 + np.sin(th) * Y0
    Yr = -np.sin(th) * X0 + np.cos(th) * Y0
    return Z + A * np.exp(-(Xr**2 / (2*sx**2) + Yr**2 / (2*sy**2)))


def mm_to_px(gt, L, N):
    px_per_mm = (N - 1) / L
    return [(round(r.yc_mm * px_per_mm), round(r.xc_mm * px_per_mm))
            for _, r in gt.iterrows()]


# ============ SINTÉTICO ============
N = int(round(L / STEP)) + 1
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, 5.0)
for _, r in gt.iterrows():
    Z = add_rotated_gaussian(Z, X, Y, r.xc_mm, r.yc_mm,
                             r.sigma1_mm, r.sigma2_mm, -r.A_mm, r.theta_deg)
np.random.seed(42)
Z += np.random.normal(0, 0.01, Z.shape)

Zc = Z - 5.0  # sustracción de la base
centers_px = mm_to_px(gt, L, N)
coeffs = sg2d_jacobian_projection_cv2(Zc, WINDOW_SIZE)
est = estimate_bump_parameters(centers_px, coeffs, STEP)

rows = []
for i, ((_, r), e) in enumerate(zip(gt.iterrows(), est)):
    rows.append({'idx': r.idx,
                 'A_real': r.A_mm, 'A_est': e['A'],
                 's1_real': r.sigma1_mm, 's1_est': e['sigma1'],
                 's2_real': r.sigma2_mm, 's2_est': e['sigma2']})
df = pd.DataFrame(rows)
print("=== SINTÉTICO ===")
print(df.to_string(index=False))
for col, lab in [('A', 'A (mm)'), ('s1', 'sigma1 (mm)'), ('s2', 'sigma2 (mm)')]:
    real = df[f'{col}_real']; estv = df[f'{col}_est']
    rho = np.corrcoef(real, estv)[0, 1]
    mae = np.mean(np.abs(real - estv))
    print(f'{lab}: rho={rho:.3f}  MAE={mae:.3f} mm')

# ============ REAL (placeholder) ============
print("\n=== REAL (placeholder) ===")
PATH_SCAN = Path('/home/mario-ross/Proyectos/Tesis/python_pruebas/escaneos/escaneo_150mm.npy')
PATH_CTR = Path('/home/mario-ross/Proyectos/Tesis/python_pruebas/escaneos/centros_150mm.csv')
if PATH_SCAN.exists() and PATH_CTR.exists():
    Z_real = np.load(PATH_SCAN)
    ctr = pd.read_csv(PATH_CTR)
    step_real = L / (Z_real.shape[0] - 1)
    Zc_real = Z_real - np.median(Z_real)  # base robusta (mediana)
    centers_real = [(int(r.y_px), int(r.x_px)) for _, r in ctr.iterrows()]
    coeffs_r = sg2d_jacobian_projection_cv2(Zc_real, WINDOW_SIZE)
    est_r = estimate_bump_parameters(centers_real, coeffs_r, step_real)
    print(f'escaneo {Z_real.shape}, step={step_real:.3f} mm/px')
    for i, ((_, r), e) in enumerate(zip(gt.iterrows(), est_r)):
        print(f'idx={r.idx}: A {r.A_mm:.2f}->{e["A"]:.2f}  s1 {r.sigma1_mm:.2f}->{e["sigma1"]:.2f}')
else:
    print('AVISO: no están los archivos de escaneo. Se rellenan RUTA_ESCANEO y RUTA_CENTROS.')

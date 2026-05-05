"""
gear_defect_pipeline.py
=======================
Pipeline completo de análisis de defectos en piezas cilíndricas (engranajes).

Función principal:
    analyze_defect(sample, base_dir, **kwargs) -> dict

Devuelve un diccionario con todas las métricas del defecto:
    largo, ancho, ang_deg, h_max, h_min, V_pos, V_neg, V_net,
    K_best_pos, K_best_neg, Z_diff_std, pix_mm, ...

Uso mínimo:
    from gear_defect_pipeline import analyze_defect
    result = analyze_defect("151225_160136_P44512_S2_7761", "../gears_defectos_xyz_etiquetados")
"""

import os, sys
import numpy as np
import cv2
from scipy.ndimage import median_filter, distance_transform_edt, gaussian_filter1d
from scipy.signal import find_peaks
from scipy.interpolate import griddata
from scipy.optimize import least_squares
import time

# ── Parámetros por defecto ────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    erosion_px      = 5,       # erosión de la máscara YOLO (px)
    med_win         = 15,      # ventana mediana (px)
    ring_mm         = 8.0,     # anchura anillo inpainting (mm)
    k_max           = 12,      # número máximo de splats
    win_defect_mm   = 4.0,     # ventana SG local para inicializar splat (mm)
    lambda_pen      = 5.0,     # penalización splat fuera de máscara
    thr_frac        = 0.05,    # umbral envolvente splats para OBB
    n_splat_sub     = 500,     # submuestreo observaciones splat
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_xyz(sample: str, base_dir: str):
    """Carga la imagen XYZ calibrada. Devuelve (X, Y, Z, H, W, valid)."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import camera.raw as raw

    entities = os.path.join(base_dir, sample, "debug", "entities")
    xyz = raw.read_img_raw(os.path.join(entities, "save_xyz", f"{sample}_save_xyz.raw"))
    H, W = xyz.shape[:2]
    X, Y, Z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    valid = np.isfinite(Z) & (Z != 0)
    return X, Y, Z, H, W, valid, entities


def load_defect_mask(sample: str, entities: str, H: int, W: int, erosion_px: int):
    """Carga y erosiona la máscara de segmentación YOLO.
    Devuelve (polygons, defect_mask_raw, defect_mask_eroded)."""
    txt_path = os.path.join(entities, "rgb_seg", f"{sample}_rgb_seg.txt")
    mask_raw = np.zeros((H, W), dtype=np.uint8)
    polygons = []
    with open(txt_path) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 5:
                continue
            coords = np.array(vals[1:], dtype=float).reshape(-1, 2)
            pts_px = (coords * np.array([W, H])).astype(np.int32)
            polygons.append(pts_px)
            cv2.fillPoly(mask_raw, [pts_px], 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1))
    mask_eroded = cv2.erode(mask_raw, ker)
    return polygons, mask_raw, mask_eroded


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESADO
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_z(Z, valid, med_win: int):
    """NN fill + mediana. Devuelve Z_smooth."""
    fill_idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    Z_filled = Z[tuple(fill_idx)]
    Z_med = median_filter(Z_filled.astype(np.float64), size=med_win)
    return np.where(valid, Z.astype(float), Z_med)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NORMALES SG 2D
# ═══════════════════════════════════════════════════════════════════════════════

def sg2d_normals(Z_smooth, valid, X, Y, H, W, sg_win_px: int):
    """SG 2D paraboloide → normales (Nx, Ny, Nz), θ cenital, φ azimutal, pix_mm."""
    dx = float((X[valid].max() - X[valid].min()) / W)
    dy = float((Y[valid].max() - Y[valid].min()) / H)
    pix_mm = (dx + dy) / 2

    hw = sg_win_px // 2
    yi, xi = np.mgrid[-hw:hw + 1, -hw:hw + 1]
    reg = np.stack([xi**2, yi**2, xi * yi, xi, yi, np.ones_like(xi)], axis=-1)
    A = reg.reshape(-1, 6)
    P = np.linalg.inv(A.T @ A) @ A.T
    ks = P.reshape(6, sg_win_px, sg_win_px)
    coefs = np.stack(
        [cv2.filter2D(Z_smooth.astype(np.float64), -1, k,
                      borderType=cv2.BORDER_REFLECT) for k in ks],
        axis=-1
    )
    Gx = coefs[..., 3] / pix_mm
    Gy = coefs[..., 4] / pix_mm
    norm_len = np.sqrt(Gx**2 + Gy**2 + 1.0)
    Nx = -Gx / norm_len
    Ny = -Gy / norm_len
    Nz = 1.0 / norm_len
    theta = np.degrees(np.arccos(np.clip(np.abs(Nz), 0, 1)))
    phi   = np.degrees(np.arctan2(Ny, Nx))
    return Nx, Ny, Nz, theta, phi, pix_mm


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SEGMENTACIÓN DE SUPERFICIES
# ═══════════════════════════════════════════════════════════════════════════════

def segment_surfaces(Z, valid, theta, phi, defect_mask_early,
                     thr_tapa_deg: float, n_wall_bins: int):
    """θ → flat/wall; histograma Z → K_flat caras; φ → sectores pared.
    Devuelve (seg_flat, seg_wall, flat_label_img, K_flat, phi_bin,
              cx_piece, cy_piece, surf_label, K_surf).
    """
    X_local = np.arange(valid.shape[1])  # no X mm, sino col (no se usa aquí)
    defect_bool = defect_mask_early.astype(bool)
    seg_flat = valid & (theta < thr_tapa_deg)  & ~defect_bool
    seg_wall = valid & (theta >= thr_tapa_deg) & ~defect_bool

    # Histograma Z → caras planas
    z_flat = Z[seg_flat]
    counts, edges = np.histogram(z_flat, bins=300)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = gaussian_filter1d(counts.astype(float), sigma=2.0)
    peaks, _ = find_peaks(smooth, height=smooth.max() * 0.05, distance=300 // 20)
    K_flat = len(peaks)
    peak_z = centers[peaks]
    dist_to_peaks = np.abs(z_flat[:, None] - peak_z[None, :])
    z_label = np.argmin(dist_to_peaks, axis=1)
    flat_label_img = np.full(valid.shape, -1, dtype=int)
    flat_label_img[seg_flat] = z_label

    # Centro pieza (media de paredes)
    Xw = np.where(seg_wall)[1].astype(float)
    Yw = np.where(seg_wall)[0].astype(float)
    cx_piece = float(Xw.mean())  # en píxeles aquí; se convierte al usar X, Y reales fuera
    cy_piece = float(Yw.mean())

    # φ bins → sectores pared
    phi_bin = ((phi + 180.0) / 360.0 * n_wall_bins).astype(int)
    phi_bin = np.clip(phi_bin, 0, n_wall_bins - 1)

    surf_label = np.full(valid.shape, -1, dtype=int)
    surf_label[seg_flat] = flat_label_img[seg_flat]
    surf_label[seg_wall] = K_flat + phi_bin[seg_wall]
    K_surf = K_flat + n_wall_bins

    return seg_flat, seg_wall, flat_label_img, K_flat, phi_bin, surf_label, K_surf


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Z_NOM POR GRIDDATA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_znom_griddata(Z, valid, defect_mask, pix_mm, ring_mm: float):
    """Interpola Z_nom desde el anillo exterior usando griddata lineal.
    Devuelve (Z_nom, Z_diff, z_def_vals, r_def, c_def, ring_px)."""
    ring_px  = int(round(ring_mm / pix_mm)) | 1
    ker_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px, ring_px))
    outer    = cv2.dilate(defect_mask, ker_ring).astype(bool)
    ring_mask = outer & ~defect_mask.astype(bool) & valid

    r_ring, c_ring = np.where(ring_mask)
    z_ring  = Z[r_ring, c_ring].astype(float)
    src_pts = np.column_stack([c_ring.astype(float), r_ring.astype(float)])

    r_def, c_def = np.where(defect_mask.astype(bool) & valid)
    dst_pts = np.column_stack([c_def.astype(float), r_def.astype(float)])

    z_interp = griddata(src_pts, z_ring, dst_pts, method='linear')
    nan_m = np.isnan(z_interp)
    if nan_m.any():
        z_near = griddata(src_pts, z_ring, dst_pts[nan_m], method='nearest')
        z_interp[nan_m] = z_near

    Z_nom = Z.copy().astype(float)
    Z_nom[r_def, c_def] = z_interp
    Z_diff = np.where(valid & defect_mask.astype(bool), Z.astype(float) - Z_nom, np.nan)
    z_def_vals = Z_diff[defect_mask.astype(bool) & valid]
    z_def_vals = z_def_vals[np.isfinite(z_def_vals)]
    return Z_nom, Z_diff, z_def_vals, r_def, c_def, ring_px


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MULTI-SPLAT
# ═══════════════════════════════════════════════════════════════════════════════

def _sg2d_local(Z_in, win):
    hw = win // 2
    yi, xi = np.mgrid[-hw:hw + 1, -hw:hw + 1]
    reg = np.stack([xi**2, yi**2, xi * yi, xi, yi, np.ones_like(xi)], axis=-1)
    A_m = reg.reshape(-1, 6)
    P   = np.linalg.inv(A_m.T @ A_m) @ A_m.T
    ks  = P.reshape(6, win, win)
    return np.stack([cv2.filter2D(Z_in.astype(np.float64), -1, k,
                                  borderType=cv2.BORDER_REFLECT) for k in ks], axis=-1)


def gauss2d_rotated(params, x, y):
    A, x0, y0, sx, sy, th = params
    c, s = np.cos(th), np.sin(th)
    u =  (x - x0) * c + (y - y0) * s
    v = -(x - x0) * s + (y - y0) * c
    return A * np.exp(-0.5 * (u**2 / sx**2 + v**2 / sy**2))


def gauss_mixture(params, x, y):
    K = len(params) // 6
    return sum(gauss2d_rotated(params[6*k:6*k+6], x, y) for k in range(K))


def _gauss_mixture_and_jac(params, x, y):
    K = len(params) // 6; N = len(x)
    out = np.zeros(N); J = np.zeros((N, 6 * K))
    for k in range(K):
        A, x0, y0, sx, sy, th = params[6*k:6*k+6]
        c, s = np.cos(th), np.sin(th)
        dx = x - x0; dy = y - y0
        u = dx*c + dy*s; v = -dx*s + dy*c
        eu2 = u**2/sx**2; ev2 = v**2/sy**2
        g = A * np.exp(-0.5*(eu2+ev2))
        out += g
        J[:, 6*k]   = g / A
        J[:, 6*k+1] = g * (u*c/sx**2 - v*s/sy**2)
        J[:, 6*k+2] = g * (u*s/sx**2 + v*c/sy**2)
        J[:, 6*k+3] = g * eu2 / sx
        J[:, 6*k+4] = g * ev2 / sy
        J[:, 6*k+5] = g * u*v*(1/sx**2 - 1/sy**2)
    return out, J


def volume_mixture(params):
    K = len(params) // 6
    return sum(params[6*k] * 2 * np.pi * params[6*k+3] * params[6*k+4] for k in range(K))


def run_multisplat(Z_source, defect_mask, valid, pix_mm,
                   k_max=12, win_defect_mm=4.0, lambda_pen=5.0,
                   n_sub=2000, verbose=False):
    """Ajuste de mezcla de gaussianas 2D sobre Z_source.
    Devuelve dict {K: {params, V, nfev, t}}.
    """
    H, W = Z_source.shape
    mask_def_valid = defect_mask.astype(bool) & valid
    rows_def_s, cols_def_s = np.where(mask_def_valid)

    X_MIN = float(cols_def_s.min()) * pix_mm;  X_MAX = float(cols_def_s.max()) * pix_mm
    Y_MIN = float(rows_def_s.min()) * pix_mm;  Y_MAX = float(rows_def_s.max()) * pix_mm
    # σ máximo por PCA de la máscara (ejes principales reales, no BB axis-aligned)
    _xy = np.column_stack([cols_def_s.astype(np.float64) * pix_mm,
                           rows_def_s.astype(np.float64) * pix_mm])
    _cov = np.cov(_xy.T)
    _eigvals = np.sort(np.linalg.eigvalsh(_cov))[::-1]  # mayor primero
    SIG_LONG  = min(float(np.sqrt(max(_eigvals[0], 1e-6))), win_defect_mm)
    SIG_SHORT = min(float(np.sqrt(max(_eigvals[1], 1e-6))), win_defect_mm)

    x_obs = cols_def_s.astype(np.float64) * pix_mm
    y_obs = rows_def_s.astype(np.float64) * pix_mm

    dist_px = distance_transform_edt(~mask_def_valid).astype(np.float32)
    grad_r_d, grad_c_d = np.gradient(dist_px.astype(np.float64))

    win_px = int(round(win_defect_mm / pix_mm)) | 1
    coefs_ph = _sg2d_local(
        np.where(np.isfinite(Z_source), np.clip(Z_source, 0, None), 0.0), win_px)
    coefs_ph_sc = np.stack([
        coefs_ph[..., 0]/pix_mm**2, coefs_ph[..., 1]/pix_mm**2,
        coefs_ph[..., 2]/pix_mm**2, coefs_ph[..., 3]/pix_mm,
        coefs_ph[..., 4]/pix_mm,    coefs_ph[..., 5]], axis=-1)

    z_obs_ = np.clip(Z_source[rows_def_s, cols_def_s].astype(np.float64), 0.0, None)
    N_obs  = len(z_obs_)
    rng    = np.random.default_rng(7)
    idx_s  = rng.choice(N_obs, min(n_sub, N_obs), replace=False)
    xs, ys, zs = x_obs[idx_s], y_obs[idx_s], z_obs_[idx_s]

    A_max = max(float(z_obs_.max()) * 3.0, 0.1)
    blo = [0, X_MIN, Y_MIN, 0.15, 0.15, -np.pi/2]
    bhi = [A_max, X_MAX, Y_MAX, SIG_LONG, SIG_SHORT, np.pi/2]

    res_all = {}; p_cur = []
    t0 = time.perf_counter()
    _bic_best = np.inf; _bic_patience = 2; _bic_no_improve = 0

    for K in range(1, k_max + 1):
        prev  = gauss_mixture(p_cur, x_obs, y_obs) if p_cur else np.zeros(N_obs)
        resid = np.clip(z_obs_ - prev, 0, None)
        res_img = np.zeros_like(Z_source)
        res_img[rows_def_s, cols_def_s] = resid
        f_ws = np.where(mask_def_valid, res_img, -np.inf)
        pr, pc = np.unravel_index(np.argmax(f_ws), f_ws.shape)
        mX = float(pc) * pix_mm;  mY = float(pr) * pix_mm

        a0, b0, c0_, d0, e0, f0 = coefs_ph_sc[pr, pc]
        Hm = np.array([[2*a0, c0_], [c0_, 2*b0]])
        try:
            dlt = -np.linalg.solve(Hm, [d0, e0])
            mX = float(np.clip(mX+dlt[0], blo[1]+1e-6, bhi[1]-1e-6))
            mY = float(np.clip(mY+dlt[1], blo[2]+1e-6, bhi[2]-1e-6))
            A0 = float(np.clip(f0, blo[0]+1e-6, bhi[0]-1e-6))
            Sm = -A0*np.linalg.inv(Hm) if A0 > 1e-6 else None
            if Sm is not None:
                ev_, evec_ = np.linalg.eigh(Sm)
                ord_ = np.argsort(ev_)[::-1]
                sx_ = float(np.clip(np.sqrt(abs(ev_[ord_[0]])), blo[3]+1e-6, bhi[3]-1e-6))
                sy_ = float(np.clip(np.sqrt(abs(ev_[ord_[1]])), blo[4]+1e-6, bhi[4]-1e-6))
                th_ = float(np.arctan2(evec_[1, ord_[0]], evec_[0, ord_[0]]))
            else:
                sx_, sy_, th_ = SIG_LONG*0.3, SIG_SHORT*0.4, 0.0
        except Exception:
            A0 = float(np.clip(res_img[pr, pc], blo[0]+1e-6, bhi[0]-1e-6))
            sx_, sy_, th_ = SIG_LONG*0.3, SIG_SHORT*0.4, 0.0

        p_new = [float(np.clip(v, lo+1e-6, hi-1e-6))
                 for v, lo, hi in zip([A0, mX, mY, sx_, sy_, th_], blo, bhi)]
        p0 = p_cur + p_new

        cf = {}
        def fun(p):
            p = np.asarray(p)
            if not np.array_equal(p, cf.get('p')):
                fv, jv = _gauss_mixture_and_jac(p, xs, ys)
                cf.update(p=p.copy(), r=fv-zs, j=jv)
            K_ = len(p)//6
            pen = np.zeros(K_)
            for k_ in range(K_):
                ci_ = int(np.clip(p[6*k_+1]/pix_mm, 0, W-1))
                ri_ = int(np.clip(p[6*k_+2]/pix_mm, 0, H-1))
                d_  = float(dist_px[ri_, ci_]) * pix_mm
                if d_ > 0:
                    pen[k_] = lambda_pen * d_
            return np.concatenate([cf['r'], pen])

        def jac(p):
            p = np.asarray(p)
            if not np.array_equal(p, cf.get('p')):
                fv, jv = _gauss_mixture_and_jac(p, xs, ys)
                cf.update(p=p.copy(), r=fv-zs, j=jv)
            K_ = len(p)//6
            Jp = np.zeros((K_, len(p)))
            for k_ in range(K_):
                ci_ = int(np.clip(p[6*k_+1]/pix_mm, 0, W-1))
                ri_ = int(np.clip(p[6*k_+2]/pix_mm, 0, H-1))
                d_  = float(dist_px[ri_, ci_]) * pix_mm
                if d_ > 0:
                    Jp[k_, 6*k_+1] = lambda_pen * float(grad_c_d[ri_, ci_])
                    Jp[k_, 6*k_+2] = lambda_pen * float(grad_r_d[ri_, ci_])
            return np.vstack([cf['j'], Jp])

        rK = least_squares(fun, p0, jac=jac, bounds=(blo*K, bhi*K),
                           method='trf', xtol=1e-4, ftol=1e-4, gtol=1e-4,
                           max_nfev=5_000)
        p_cur = list(rK.x)
        V_K   = volume_mixture(p_cur)
        t_K   = time.perf_counter() - t0
        res_all[K] = dict(params=p_cur.copy(), V=V_K, nfev=rK.nfev, t=t_K)

        # Early stopping: BIC inline sobre todas las observaciones (rápido)
        _bic_K = (np.sum((z_obs_ - gauss_mixture(p_cur, x_obs, y_obs))**2)
                  + 6 * K * float(np.var(z_obs_)))
        if _bic_K < _bic_best:
            _bic_best = _bic_K; _bic_no_improve = 0
        else:
            _bic_no_improve += 1

        if verbose:
            print(f'  K={K}: V={V_K:.4f} mm³  nfev={rK.nfev}  t={t_K*1000:.0f}ms  BIC={_bic_K:.4f}')

        if _bic_no_improve >= _bic_patience:
            if verbose:
                print(f'  → early stop en K={K} (BIC no mejora {_bic_patience} iteraciones)')
            break

    return res_all


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BIC + SELECCIÓN DE K
# ═══════════════════════════════════════════════════════════════════════════════

def select_k_bic(results_dict, Z_diff_clean, rows_def_s, cols_def_s, x_obs, y_obs, sign=1):
    """BIC para seleccionar K óptimo. Devuelve (K_best, V_best, params_best)."""
    z_obs_ = np.clip(sign * Z_diff_clean[rows_def_s, cols_def_s], 0, None)
    K_vals = sorted(results_dict.keys())
    bics = [np.sum((z_obs_ - gauss_mixture(results_dict[k]['params'], x_obs, y_obs))**2)
            + 6 * k * float(np.var(z_obs_)) for k in K_vals]
    K_best = K_vals[int(np.argmin(bics))]
    return K_best, results_dict[K_best]['V'], results_dict[K_best]['params']


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MÉTRICAS OBB
# ═══════════════════════════════════════════════════════════════════════════════

def compute_obb_metrics(p_best_pos, p_best_neg, rows_def_s, cols_def_s,
                        pix_mm, thr_frac=0.05, nc_m=300, nr_m=150):
    """OBB por PCA sobre los píxeles reales de la máscara del defecto.
    Los splats se usan solo para h_max/h_min y visualización."""
    c_m = np.linspace(float(cols_def_s.min()), float(cols_def_s.max()), nc_m)
    r_m = np.linspace(float(rows_def_s.min()), float(rows_def_s.max()), nr_m)
    CM, RM = np.meshgrid(c_m, r_m)
    xm = CM * pix_mm;  ym = RM * pix_mm

    Z_spl_pos = gauss_mixture(p_best_pos, xm.ravel(), ym.ravel()).reshape(nr_m, nc_m)
    Z_spl_neg = gauss_mixture(p_best_neg, xm.ravel(), ym.ravel()).reshape(nr_m, nc_m)

    h_max =  float(Z_spl_pos.max())
    h_min = -float(Z_spl_neg.max())

    Z_envelope  = np.maximum(Z_spl_pos, Z_spl_neg)
    mask_active = Z_envelope > thr_frac * float(Z_envelope.max())

    # OBB calculado sobre los píxeles reales de la máscara (no sobre las colas
    # del modelo Gaussiano, que se extienden hasta ~2.45σ más allá del defecto)
    pts_mm = np.column_stack([cols_def_s.astype(float) * pix_mm,
                               rows_def_s.astype(float) * pix_mm])
    if len(pts_mm) >= 2:
        mu  = pts_mm.mean(axis=0)
        cov = np.cov((pts_mm - mu).T)
        if cov.ndim < 2:
            cov = np.eye(2) * float(cov)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        order    = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, order]
        proj     = (pts_mm - mu) @ eig_vecs
        semieje_largo = float((proj[:, 0].max() - proj[:, 0].min()) / 2)
        semieje_ancho = float((proj[:, 1].max() - proj[:, 1].min()) / 2)
        largo  = 2 * semieje_largo
        ancho  = 2 * semieje_ancho
        ang_deg = float(np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])))
    else:
        largo = ancho = semieje_largo = semieje_ancho = ang_deg = 0.0
        mu = np.zeros(2);  eig_vecs = np.eye(2)

    return dict(
        largo=largo, ancho=ancho, ang_deg=ang_deg,
        semieje_largo=semieje_largo, semieje_ancho=semieje_ancho,
        h_max=h_max, h_min=abs(h_min),
        mu=mu, eig_vecs=eig_vecs,
        Z_spl_pos=Z_spl_pos, Z_spl_neg=Z_spl_neg,
        Z_envelope=Z_envelope, mask_active=mask_active,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FUNCIONES PRINCIPALES
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_sample(sample: str, base_dir: str, **kwargs) -> dict:
    """Carga y preprocesa una muestra completa (XYZ + relleno + mediana).

    Se ejecuta UNA SOLA VEZ por muestra aunque haya varios defectos.
    Devuelve un dict con los arrays compartidos necesarios para analizar
    cada defecto individualmente con analyze_polygon().
    """
    p = {**DEFAULT_PARAMS, **kwargs}

    # ── 1. Carga ──────────────────────────────────────────────────────────────
    X, Y, Z, H, W, valid, entities = load_xyz(sample, base_dir)
    polygons, defect_mask_raw, _ = load_defect_mask(
        sample, entities, H, W, p['erosion_px'])

    # ── 2. Preprocesado ───────────────────────────────────────────────────────
    Z_smooth = preprocess_z(Z, valid, p['med_win'])

    # pix_mm: resolución espacial (mm/px) desde el rango métrico de X e Y
    dx = float((X[valid].max() - X[valid].min()) / W)
    dy = float((Y[valid].max() - Y[valid].min()) / H)
    pix_mm = (dx + dy) / 2

    return dict(
        sample   = sample,
        entities = entities,
        X=X, Y=Y, Z=Z, H=H, W=W,
        valid    = valid,
        Z_smooth = Z_smooth,
        pix_mm   = pix_mm,
        polygons = polygons,
        params   = p,
    )


def analyze_polygon(polygon: np.ndarray, shared: dict, defect_idx: int = 0,
                    verbose: bool = False) -> dict:
    """Analiza UN único polígono de defecto usando los datos compartidos de la muestra.

    Parámetros
    ----------
    polygon    : array (N,2) con coordenadas en píxeles [col, row]
    shared     : dict devuelto por prepare_sample()
    defect_idx : índice del defecto dentro de la muestra (para identificación)
    verbose    : imprimir progreso del ajuste multi-splat
    """
    p      = shared['params']
    X      = shared['X'];   Y = shared['Y'];   Z = shared['Z']
    H      = shared['H'];   W = shared['W']
    valid  = shared['valid']
    pix_mm = shared['pix_mm']
    sample = shared['sample']

    t_start = time.perf_counter()

    # ── Máscara individual (solo este polígono, con erosión) ───────────────────
    mask_raw = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask_raw, [polygon], 1)
    ker_er = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * p['erosion_px'] + 1, 2 * p['erosion_px'] + 1))
    defect_mask = cv2.erode(mask_raw, ker_er)

    if defect_mask.sum() == 0:
        # Si la erosión elimina toda la máscara, usar la original
        defect_mask = mask_raw.copy()

    # ── Z_nom por griddata ────────────────────────────────────────────────────
    (Z_nom, Z_diff, z_def_vals,
     r_def, c_def, ring_px) = compute_znom_griddata(
        Z, valid, defect_mask, pix_mm, p['ring_mm'])

    Z_diff_clean = np.nan_to_num(Z_diff, nan=0.0)

    mask_def_valid = defect_mask.astype(bool) & valid
    rows_def_s, cols_def_s = np.where(mask_def_valid)
    x_obs = cols_def_s.astype(np.float64) * pix_mm
    y_obs = rows_def_s.astype(np.float64) * pix_mm

    tag = f"{sample}[{defect_idx}]"

    # ── Multi-splat ───────────────────────────────────────────────────────────
    if verbose:
        print(f"[{tag}] Multi-splat POSITIVO …")
    results_pos = run_multisplat(
        Z_diff_clean, defect_mask, valid, pix_mm,
        k_max=p['k_max'], win_defect_mm=p['win_defect_mm'],
        lambda_pen=p['lambda_pen'], n_sub=p['n_splat_sub'], verbose=verbose)

    if verbose:
        print(f"[{tag}] Multi-splat NEGATIVO …")
    results_neg = run_multisplat(
        -Z_diff_clean, defect_mask, valid, pix_mm,
        k_max=p['k_max'], win_defect_mm=p['win_defect_mm'],
        lambda_pen=p['lambda_pen'], n_sub=p['n_splat_sub'], verbose=verbose)

    # ── BIC → K óptimo ───────────────────────────────────────────────────────
    K_best_pos, V_pos, p_best_pos = select_k_bic(
        results_pos, Z_diff_clean, rows_def_s, cols_def_s, x_obs, y_obs, sign=+1)
    K_best_neg, V_neg, p_best_neg = select_k_bic(
        results_neg, Z_diff_clean, rows_def_s, cols_def_s, x_obs, y_obs, sign=-1)

    # ── Métricas OBB ─────────────────────────────────────────────────────────
    obb = compute_obb_metrics(
        p_best_pos, p_best_neg, rows_def_s, cols_def_s,
        pix_mm, thr_frac=p['thr_frac'])

    t_total = time.perf_counter() - t_start
    if verbose:
        print(f"[{tag}] Completado en {t_total:.1f}s")

    return dict(
        sample      = sample,
        defect_idx  = defect_idx,
        pix_mm      = pix_mm,
        n_defect_px = int(mask_def_valid.sum()),
        Z_diff_std  = float(np.nanstd(z_def_vals))  if len(z_def_vals) else float('nan'),
        Z_diff_min  = float(np.nanmin(z_def_vals))  if len(z_def_vals) else float('nan'),
        Z_diff_max  = float(np.nanmax(z_def_vals))  if len(z_def_vals) else float('nan'),
        K_best_pos  = K_best_pos,
        V_pos       = float(V_pos),
        K_best_neg  = K_best_neg,
        V_neg       = float(V_neg),
        V_net       = float(V_pos - V_neg),
        params_pos  = p_best_pos,
        params_neg  = p_best_neg,
        results_pos = results_pos,
        results_neg = results_neg,
        largo       = obb['largo'],
        ancho       = obb['ancho'],
        ang_deg     = obb['ang_deg'],
        h_max       = obb['h_max'],
        h_min       = obb['h_min'],
        Z_diff      = Z_diff,
        Z_nom       = Z_nom,
        polygon     = polygon,           # polígono de este defecto concreto
        rows_def_s  = rows_def_s,
        cols_def_s  = cols_def_s,
        x_obs       = x_obs,
        y_obs       = y_obs,
        obb         = obb,
        t_total     = t_total,
    )


def analyze_defect(sample: str, base_dir: str, verbose=False, **kwargs) -> list:
    """Pipeline completo: prepara la muestra y analiza CADA defecto por separado.

    Devuelve una lista de dicts, uno por polígono/defecto encontrado.
    Si solo hay un defecto, la lista tiene un único elemento.
    """
    shared = prepare_sample(sample, base_dir, **kwargs)
    polygons = shared['polygons']

    if not polygons:
        raise ValueError(f"No se encontraron polígonos de defecto en {sample}")

    results = []
    for i, poly in enumerate(polygons):
        if verbose:
            print(f"\n── Defecto {i+1}/{len(polygons)} ──")
        res = analyze_polygon(poly, shared, defect_idx=i, verbose=verbose)
        results.append(res)

    return results

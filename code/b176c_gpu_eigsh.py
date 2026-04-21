#!/usr/bin/env python3
"""B176c -- GPU-versnelde eigsh voor CGAL-SDP op n >= 5000.

Doel
----
B176b-CGAL haalt n ~ 2000 op een laptop-CPU. Voor n >= 5000 is de
per-iteratie-eigsh-call (scipy.sparse.linalg.eigsh, ARPACK-Lanczos) de
bottleneck: enkele tientallen minuten per call. Dit module biedt drop-in
vervanging met vier backends, waarbij CuPy-pathen de matvec op GPU
draaien:

  * ``cupy_lobpcg``   -- cupyx.scipy.sparse.linalg.lobpcg (block-iterative,
                         natuurlijk warm-startbaar via initial guess X0)
  * ``cupy_lanczos``  -- cupyx.scipy.sparse.linalg.eigsh (Lanczos,
                         warm-start via v0-startvector)
  * ``scipy_lobpcg``  -- scipy.sparse.linalg.lobpcg (CPU, warm-startbaar)
  * ``scipy_arpack``  -- scipy.sparse.linalg.eigsh (huidige B176b-default)

De verwachte winst komt uit drie richtingen:

  1. **GPU-matvec** voor de dominante O(iter x nnz) kosten.
  2. **Warm-start**: v_{k-1} is dicht bij v_k; LOBPCG/Lanczos convergeren
     dan in 2-5 matvecs i.p.v. 30-80.
  3. **Dual-eigsh via LMO-hergebruik** (bonus): y_k ~ z_k laat de
     dual_upper_bound-eigvec vaak warm-starten met v_{k-1} van de LMO.

API-ontwerp
-----------
Centrale functie::

    res = gpu_eigsh_smallest(
        L, z,
        coef_L=-0.25,
        v0=None,                 # warm-start
        tol=1e-8,
        backend='auto',
    )
    # res.v : np.ndarray (n,)   -- eigenvector op CPU
    # res.lam : float            -- kleinste eigenwaarde
    # res.info : dict            -- {backend, n_matvec, wall_time, ...}

De operator G = coef_L * L + diag(z) wordt intern gebouwd per backend.
Voor CuPy-backends wordt L 1x geupload en in GPU-memory gehouden zolang
de caller dezelfde L-referentie hergebruikt (via id()-cache).

Integratie
----------
In ``b176b_cgal_sdp.cgal_maxcut_sdp`` en ``b176_frank_wolfe_sdp.
lmo_spectraplex`` kan de eigsh-call vervangen worden door::

    from b176c_gpu_eigsh import gpu_eigsh_smallest
    res = gpu_eigsh_smallest(L, z, v0=v_prev, backend='auto')
    v_prev = res.v
    lam_min = res.lam

Warm-start-volgorde:
  * LMO-iter k:        v0 = v_{k-1} (previous LMO-eigvec)
  * dual UB iter k:    v0 = v_LMO_k (de LMO heeft net dezelfde structuur
                       opgelost; y_k ~ z_k bij convergentie)

Referenties
-----------
* Knyazev (2001). "Toward the optimal preconditioned eigensolver:
  LOBPCG." SIAM J. Sci. Comput.
* Saad (2011). "Numerical methods for large eigenvalue problems," 2nd ed.
* CuPy-docs: https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.lobpcg.html
* Yurtsever, Fercoq, Cevher (2019). ICML.

Sandbox-caveat
--------------
Dit module is geschreven om op een laptop met CUDA + CuPy te draaien.
In de ZornQ-sandbox (geen GPU) zijn alleen de scipy-backends geverifieerd;
de CuPy-pathen zijn achter ``try: import cupy``-guards met graceful
fallback, en de correctness-tests voor die pathen skip als CuPy afwezig.
De benchmark-script (``b176c_benchmark.py``) draait alle vier als mogelijk
en produceert een CSV + plot.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ============================================================
# CuPy-availability detectie
# ============================================================

try:
    import cupy as _cp  # type: ignore
    import cupyx.scipy.sparse as _cpsp  # type: ignore
    import cupyx.scipy.sparse.linalg as _cpspla  # type: ignore
    _CUPY_AVAILABLE = True
    try:
        # Force-init: raises if no CUDA device.
        _cp.cuda.Device(0).compute_capability  # noqa: B018
        _CUDA_AVAILABLE = True
    except Exception:  # pragma: no cover
        _CUDA_AVAILABLE = False
except Exception:  # pragma: no cover
    _cp = None  # type: ignore
    _cpsp = None  # type: ignore
    _cpspla = None  # type: ignore
    _CUPY_AVAILABLE = False
    _CUDA_AVAILABLE = False


def available_backends() -> list[str]:
    """Return list of backends that can run in this process.

    ``scipy_arpack`` en ``scipy_lobpcg`` zijn altijd beschikbaar.
    ``cupy_lanczos`` / ``cupy_lobpcg`` alleen als CuPy + CUDA werken.
    """
    out = ["scipy_arpack", "scipy_lobpcg"]
    if _CUPY_AVAILABLE and _CUDA_AVAILABLE:
        out.extend(["cupy_lanczos", "cupy_lobpcg"])
    return out


def cupy_available() -> bool:
    """Return True iff CuPy-backends kunnen draaien."""
    return _CUPY_AVAILABLE and _CUDA_AVAILABLE


# ============================================================
# GPU-L cache (id(L) -> cupy-CSR) om re-upload te vermijden
# ============================================================

_GPU_L_CACHE: dict[int, Any] = {}


def _ensure_gpu_L(L: sp.spmatrix):
    """Upload L naar GPU (CSR) en cache op Python-id van L.

    Belt niet als CuPy ontbreekt. Caller moet dat zelf checken.
    """
    if not cupy_available():
        raise RuntimeError("CuPy/CUDA niet beschikbaar")
    key = id(L)
    if key not in _GPU_L_CACHE:
        L_coo = L.tocoo()
        row = _cp.asarray(L_coo.row, dtype=_cp.int32)
        col = _cp.asarray(L_coo.col, dtype=_cp.int32)
        data = _cp.asarray(L_coo.data, dtype=_cp.float64)
        L_gpu = _cpsp.coo_matrix(
            (data, (row, col)), shape=L.shape
        ).tocsr()
        _GPU_L_CACHE[key] = L_gpu
    return _GPU_L_CACHE[key]


def clear_gpu_cache() -> None:
    """Leeg de GPU-L-cache (roep aan tussen benchmark-runs)."""
    _GPU_L_CACHE.clear()
    if cupy_available():
        _cp.get_default_memory_pool().free_all_blocks()


# ============================================================
# Resultaat-container
# ============================================================


@dataclass
class EigshResult:
    v: np.ndarray          # eigenvector op CPU (shape (n,))
    lam: float             # eigenvalue
    info: dict = field(default_factory=dict)


# ============================================================
# Dense-fallback (voor kleine n)
# ============================================================


def _dense_smallest(L: sp.spmatrix, z: np.ndarray, coef_L: float) -> EigshResult:
    """Exact via numpy.linalg.eigh op de dense G-matrix."""
    n = L.shape[0]
    t0 = time.perf_counter()
    G = coef_L * L.toarray() + np.diag(z)
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    lam = float(w[0])
    v = V[:, 0].copy()
    return EigshResult(
        v=v, lam=lam,
        info={"backend": "dense_eigh", "n": n,
              "wall_time": time.perf_counter() - t0,
              "n_matvec": None},
    )


# ============================================================
# Backend 1: scipy ARPACK (baseline, reproduceert B176b-gedrag)
# ============================================================


def _scipy_arpack(
    L: sp.spmatrix, z: np.ndarray, coef_L: float,
    v0: Optional[np.ndarray], tol: float, maxiter: Optional[int],
) -> EigshResult:
    n = L.shape[0]
    if maxiter is None:
        maxiter = max(2000, min(10000, n * 2))

    matvec_count = [0]

    def mv(x: np.ndarray) -> np.ndarray:
        matvec_count[0] += 1
        return coef_L * L.dot(x) + z * x

    op = spla.LinearOperator((n, n), matvec=mv, dtype=float)
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = dict(k=1, which="SA", tol=tol, maxiter=maxiter)
    if v0 is not None:
        kwargs["v0"] = np.asarray(v0, dtype=float)

    try:
        w, V = spla.eigsh(op, **kwargs)
        lam = float(w[0])
        v = V[:, 0].copy()
        ok = True
        detail = "converged"
    except spla.ArpackNoConvergence as exc:
        if exc.eigenvalues.size > 0:
            idx = int(np.argmin(exc.eigenvalues))
            lam = float(exc.eigenvalues[idx])
            v = exc.eigenvectors[:, idx].copy()
            ok = False
            detail = "partial"
        else:
            rng = np.random.default_rng(42)
            v = rng.standard_normal(n)
            v /= np.linalg.norm(v) + 1e-18
            lam = float(v @ mv(v))
            ok = False
            detail = "random_rayleigh"

    return EigshResult(
        v=v, lam=lam,
        info={"backend": "scipy_arpack", "n": n,
              "wall_time": time.perf_counter() - t0,
              "n_matvec": matvec_count[0],
              "converged": ok, "detail": detail,
              "warm_start": v0 is not None},
    )


# ============================================================
# Backend 2: scipy LOBPCG (warm-startbaar op CPU)
# ============================================================


def _scipy_lobpcg(
    L: sp.spmatrix, z: np.ndarray, coef_L: float,
    v0: Optional[np.ndarray], tol: float, maxiter: Optional[int],
) -> EigshResult:
    n = L.shape[0]
    if maxiter is None:
        maxiter = max(200, min(2000, n // 4))

    # LOBPCG zoekt grootste eigenwaardes (largest=True) of kleinste (largest=False).
    # We willen lambda_min -> largest=False.
    if v0 is None:
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((n, 1))
    else:
        X0 = np.asarray(v0, dtype=float).reshape(n, 1).copy()
    # LOBPCG-norm
    X0 /= np.linalg.norm(X0, axis=0, keepdims=True) + 1e-18

    matvec_count = [0]

    def mv(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            matvec_count[0] += 1
            return coef_L * L.dot(X) + z * X
        matvec_count[0] += X.shape[1]
        return coef_L * L.dot(X) + z[:, None] * X

    op = spla.LinearOperator((n, n), matvec=mv, dtype=float)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, V = spla.lobpcg(op, X0, tol=tol, maxiter=maxiter, largest=False)
    lam = float(np.atleast_1d(w)[0])
    v = np.atleast_2d(V.T).T[:, 0].copy()

    return EigshResult(
        v=v, lam=lam,
        info={"backend": "scipy_lobpcg", "n": n,
              "wall_time": time.perf_counter() - t0,
              "n_matvec": matvec_count[0],
              "warm_start": v0 is not None},
    )


# ============================================================
# Backend 3: CuPy Lanczos (cupyx.scipy.sparse.linalg.eigsh)
# ============================================================


def _cupy_lanczos(
    L: sp.spmatrix, z: np.ndarray, coef_L: float,
    v0: Optional[np.ndarray], tol: float, maxiter: Optional[int],
) -> EigshResult:
    if not cupy_available():  # pragma: no cover
        raise RuntimeError("CuPy/CUDA niet beschikbaar")

    n = L.shape[0]
    if maxiter is None:
        maxiter = max(2000, min(10000, n * 2))

    L_gpu = _ensure_gpu_L(L)
    z_gpu = _cp.asarray(z, dtype=_cp.float64)

    # cupyx.scipy.sparse.linalg.eigsh ondersteunt AX-matvec via sparse L,
    # maar geen diagonal shift. Bouw explicit operator:
    # We moeten kijken of cupyx LinearOperator bestaat.
    from cupyx.scipy.sparse.linalg import LinearOperator as _CPLinOp  # type: ignore

    matvec_count = [0]

    def mv(x):
        matvec_count[0] += 1
        return coef_L * L_gpu.dot(x) + z_gpu * x

    op = _CPLinOp((n, n), matvec=mv, dtype=_cp.float64)

    kwargs: dict[str, Any] = dict(k=1, which="SA", tol=tol, maxiter=maxiter)
    if v0 is not None:
        kwargs["v0"] = _cp.asarray(v0, dtype=_cp.float64)

    t0 = time.perf_counter()
    w, V = _cpspla.eigsh(op, **kwargs)
    _cp.cuda.Stream.null.synchronize()
    wall = time.perf_counter() - t0

    lam = float(_cp.asnumpy(w)[0])
    v = _cp.asnumpy(V[:, 0]).copy()

    return EigshResult(
        v=v, lam=lam,
        info={"backend": "cupy_lanczos", "n": n,
              "wall_time": wall, "n_matvec": matvec_count[0],
              "warm_start": v0 is not None},
    )


# ============================================================
# Backend 4: CuPy LOBPCG (block-iterative, natuurlijk warm-start)
# ============================================================


def _cupy_lobpcg(
    L: sp.spmatrix, z: np.ndarray, coef_L: float,
    v0: Optional[np.ndarray], tol: float, maxiter: Optional[int],
) -> EigshResult:
    if not cupy_available():  # pragma: no cover
        raise RuntimeError("CuPy/CUDA niet beschikbaar")

    n = L.shape[0]
    if maxiter is None:
        maxiter = max(200, min(2000, n // 4))

    L_gpu = _ensure_gpu_L(L)
    z_gpu = _cp.asarray(z, dtype=_cp.float64)

    from cupyx.scipy.sparse.linalg import LinearOperator as _CPLinOp  # type: ignore

    matvec_count = [0]

    def mv(X):
        if X.ndim == 1:
            matvec_count[0] += 1
            return coef_L * L_gpu.dot(X) + z_gpu * X
        matvec_count[0] += int(X.shape[1])
        return coef_L * L_gpu.dot(X) + z_gpu[:, None] * X

    op = _CPLinOp((n, n), matvec=mv, dtype=_cp.float64)

    if v0 is None:
        rng = _cp.random.default_rng(0)
        X0 = rng.standard_normal((n, 1), dtype=_cp.float64)
    else:
        X0 = _cp.asarray(v0, dtype=_cp.float64).reshape(n, 1).copy()
    X0 /= _cp.linalg.norm(X0, axis=0, keepdims=True) + 1e-18

    t0 = time.perf_counter()
    w, V = _cpspla.lobpcg(op, X0, tol=tol, maxiter=maxiter, largest=False)
    _cp.cuda.Stream.null.synchronize()
    wall = time.perf_counter() - t0

    lam = float(_cp.asnumpy(_cp.atleast_1d(w))[0])
    v = _cp.asnumpy(V[:, 0]).copy()

    return EigshResult(
        v=v, lam=lam,
        info={"backend": "cupy_lobpcg", "n": n,
              "wall_time": wall, "n_matvec": matvec_count[0],
              "warm_start": v0 is not None},
    )


# ============================================================
# Publieke API: gpu_eigsh_smallest
# ============================================================


def gpu_eigsh_smallest(
    L: sp.spmatrix,
    z: np.ndarray,
    coef_L: float = -0.25,
    v0: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    maxiter: Optional[int] = None,
    backend: str = "auto",
    dense_fallback_below: int = 40,
    gpu_threshold: int = 1000,
) -> EigshResult:
    """Bereken (v, lam) met G*v = lam*v, lam = lambda_min(G).

    G = coef_L * L + diag(z), zoals gebruikt in B176 en B176b.

    Parameters
    ----------
    L : scipy.sparse matrix (n x n)
        Meestal graph_laplacian(graph); CSR-vorm werkt het best.
    z : np.ndarray (n,)
        Diagonaal-shift. In B176b: z = y + beta * d.
        In dual_upper_bound: z = y.
    coef_L : float
        Scaling op L (-0.25 voor MaxCut-SDP).
    v0 : np.ndarray of None
        Warm-start startvector. None = willekeurig of backend-default.
    tol : float
        Convergentie-tolerantie. 1e-8 is B176b-default.
    maxiter : int or None
        Max iterations; None = heuristiek per backend.
    backend : str
        'auto', 'scipy_arpack', 'scipy_lobpcg', 'cupy_lanczos', 'cupy_lobpcg',
        of 'dense'. Bij 'auto':
            n <= dense_fallback_below             -> 'dense'
            cupy_available() en n >= gpu_threshold -> 'cupy_lobpcg'
            anders                                  -> 'scipy_arpack'
    dense_fallback_below : int
        Onder deze n altijd dense via numpy.linalg.eigh.
    gpu_threshold : int
        Minimale n om GPU-backends te overwegen in 'auto'-mode.

    Returns
    -------
    EigshResult met v (CPU numpy-array), lam (float), info-dict.
    """
    n = L.shape[0]
    if n != L.shape[1] or z.shape != (n,):
        raise ValueError(
            f"Shape-mismatch: L is {L.shape}, z is {z.shape}, expected ({n},)"
        )

    if n <= dense_fallback_below:
        return _dense_smallest(L, z, coef_L)

    if backend == "auto":
        if cupy_available() and n >= gpu_threshold:
            backend = "cupy_lobpcg"
        else:
            backend = "scipy_arpack"

    if backend == "dense":
        return _dense_smallest(L, z, coef_L)
    if backend == "scipy_arpack":
        return _scipy_arpack(L, z, coef_L, v0, tol, maxiter)
    if backend == "scipy_lobpcg":
        return _scipy_lobpcg(L, z, coef_L, v0, tol, maxiter)
    if backend == "cupy_lanczos":
        if not cupy_available():
            # Fallback instead of crashing
            warnings.warn("CuPy unavailable, fallback to scipy_arpack",
                          RuntimeWarning, stacklevel=2)
            return _scipy_arpack(L, z, coef_L, v0, tol, maxiter)
        return _cupy_lanczos(L, z, coef_L, v0, tol, maxiter)
    if backend == "cupy_lobpcg":
        if not cupy_available():
            warnings.warn("CuPy unavailable, fallback to scipy_lobpcg",
                          RuntimeWarning, stacklevel=2)
            return _scipy_lobpcg(L, z, coef_L, v0, tol, maxiter)
        return _cupy_lobpcg(L, z, coef_L, v0, tol, maxiter)

    raise ValueError(f"Unknown backend: {backend!r}. "
                     f"Valid: auto | dense | {' | '.join(available_backends())}")


# ============================================================
# Integratie-helper voor B176/B176b: matvec-vorm (backwards-compat)
# ============================================================


def lmo_spectraplex_warm(
    L: sp.spmatrix,
    z: np.ndarray,
    coef_L: float = -0.25,
    v_prev: Optional[np.ndarray] = None,
    backend: str = "auto",
    tol: float = 1e-8,
    dense_fallback_below: int = 40,
) -> tuple[np.ndarray, float, dict]:
    """Drop-in vervanging voor b176_frank_wolfe_sdp.lmo_spectraplex.

    Returns (v, lam_min, info).  Past warm-start toe via v_prev.
    """
    res = gpu_eigsh_smallest(
        L, z, coef_L=coef_L, v0=v_prev, tol=tol, backend=backend,
        dense_fallback_below=dense_fallback_below,
    )
    return res.v, res.lam, res.info


# ============================================================
# CLI-self-test
# ============================================================


def _selftest() -> None:  # pragma: no cover
    """Korte smoke-test: bouw random sparse L + z, test alle backends."""
    rng = np.random.default_rng(1)
    n = 200
    density = 0.01
    A = sp.random(n, n, density=density, random_state=rng, dtype=float)
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg) - A
    L = L.tocsr()
    z = rng.standard_normal(n) * 0.1

    # Ground truth
    gt = _dense_smallest(L, z, coef_L=-0.25)
    print(f"[dense]         lam = {gt.lam:+.6f}  wall = {gt.info['wall_time']*1e3:.2f}ms")

    for bk in ["scipy_arpack", "scipy_lobpcg"]:
        r = gpu_eigsh_smallest(L, z, backend=bk)
        err = abs(r.lam - gt.lam)
        print(f"[{bk:16s}] lam = {r.lam:+.6f}  err = {err:.2e}  "
              f"nmv={r.info.get('n_matvec')}  wall={r.info['wall_time']*1e3:.2f}ms")

    if cupy_available():
        for bk in ["cupy_lanczos", "cupy_lobpcg"]:
            r = gpu_eigsh_smallest(L, z, backend=bk)
            err = abs(r.lam - gt.lam)
            print(f"[{bk:16s}] lam = {r.lam:+.6f}  err = {err:.2e}  "
                  f"nmv={r.info.get('n_matvec')}  wall={r.info['wall_time']*1e3:.2f}ms")
    else:
        print("[cupy]          niet beschikbaar in deze omgeving")

    # Warm-start demonstratie
    print("\n=== Warm-start demo (scipy_lobpcg) ===")
    v_prev = None
    for step in range(3):
        # Drift z lichtjes
        z_k = z + 0.01 * step * rng.standard_normal(n)
        r = gpu_eigsh_smallest(L, z_k, v0=v_prev, backend="scipy_lobpcg")
        print(f"step {step}: lam = {r.lam:+.6f}  "
              f"nmv={r.info.get('n_matvec'):4d}  warm={r.info['warm_start']}")
        v_prev = r.v


if __name__ == "__main__":  # pragma: no cover
    _selftest()

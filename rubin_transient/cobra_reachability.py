"""Exact Subaru/PFS SKY-fiber cobra reachability using pfs_utils and ics_cobraOps."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pfs.instdata
from erfa import ErfaWarning
from pfs.datamodel import FiberStatus, PfsDesign, TargetType

DEFAULT_BLACK_DOT_MARGIN = 1.65

_CACHED_BENCH = None
_CACHED_BENCH_MARGIN: float | None = None
_TEMP_COACH_DIR: str | None = None


def _ensure_pfs_instdata_env() -> str:
    instdata_dir = os.path.dirname(pfs.instdata.__file__)
    os.environ["PFS_INSTDATA_DIR"] = instdata_dir
    return instdata_dir


def get_bench(black_dot_radius_margin: float = DEFAULT_BLACK_DOT_MARGIN):
    """Initialize (and cache) the PFS focal plane Bench with CobraCoach."""
    global _CACHED_BENCH, _CACHED_BENCH_MARGIN, _TEMP_COACH_DIR
    if _CACHED_BENCH is not None and _CACHED_BENCH_MARGIN == black_dot_radius_margin:
        return _CACHED_BENCH

    _ensure_pfs_instdata_env()
    from ics.cobraCharmer.cobraCoach.cobraCoach import CobraCoach
    from ics.cobraOps.Bench import Bench

    if _TEMP_COACH_DIR is None or not os.path.isdir(_TEMP_COACH_DIR):
        _TEMP_COACH_DIR = tempfile.mkdtemp(prefix="pfs_cobracoach_")
        atexit.register(lambda d=_TEMP_COACH_DIR: shutil.rmtree(d, ignore_errors=True))

    cobra_coach = CobraCoach(
        loadModel=True,
        trajectoryMode=True,
        rootDir=_TEMP_COACH_DIR,
    )
    bench = Bench(cobra_coach, blackDotsMargin=black_dot_radius_margin)
    _CACHED_BENCH = bench
    _CACHED_BENCH_MARGIN = black_dot_radius_margin
    return bench


def _default_obs_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sky_to_pfi(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    design: PfsDesign,
    obs_time: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sky (RA, Dec) in degrees to PFI focal plane (x, y) in mm."""
    ra_arr = np.asarray(ra_deg, dtype=float)
    dec_arr = np.asarray(dec_deg, dtype=float)
    if ra_arr.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    _ensure_pfs_instdata_env()
    from pfs.utils.coordinates.CoordTransp import CoordinateTransform

    otime = obs_time or _default_obs_time()
    xy = np.vstack([ra_arr, dec_arr])
    cent = np.array([[float(design.raBoresight)], [float(design.decBoresight)]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        pfi = CoordinateTransform(
            xy,
            mode="sky_pfi",
            pa=float(design.posAng),
            cent=cent,
            time=otime,
        )
    return np.asarray(pfi[0], dtype=float), np.asarray(pfi[1], dtype=float)


def check_sky_fiber_reachability(
    design: PfsDesign,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    bench=None,
    obs_time: str | None = None,
    black_dot_radius_margin: float = DEFAULT_BLACK_DOT_MARGIN,
) -> pd.DataFrame:
    """Evaluate which targets can be reached by a GOOD SKY fiber cobra in ``design``.

    Checks:
    1. Sky -> PFI focal plane transform (mm) via ``pfs.utils.coordinates.CoordTransp``.
    2. Black-dot avoidance using ``bench.blackDots`` (with ``black_dot_radius_margin``).
    3. Exact cobra patrol ring (``rMin <= dist <= rMax``) for every GOOD SKY fiber
       whose cobra is marked ``isGood`` in ``bench.cobras``.
    4. Distance to the nearest currently assigned non-SKY fiber (for collision preview).
    """
    ra_arr = np.asarray(ra_deg, dtype=float)
    dec_arr = np.asarray(dec_deg, dtype=float)
    n_targets = len(ra_arr)

    if n_targets == 0:
        return pd.DataFrame(
            columns=[
                "pfi_x_mm",
                "pfi_y_mm",
                "in_black_dot",
                "sky_reachable",
                "n_sky_cobras",
                "sky_fiber_ids",
                "sky_cobra_ids",
                "nearest_assigned_mm",
            ]
        )

    if bench is None:
        bench = get_bench(black_dot_radius_margin=black_dot_radius_margin)

    pfi_x, pfi_y = sky_to_pfi(ra_arr, dec_arr, design, obs_time=obs_time)
    pos = pfi_x + 1j * pfi_y
    finite_pos = np.isfinite(pfi_x) & np.isfinite(pfi_y)

    # 1. Black-dot check
    bd_centers = np.asarray(bench.blackDots.centers, dtype=complex)
    bd_radii = np.asarray(bench.blackDots.radius, dtype=float)
    in_black_dot = np.zeros(n_targets, dtype=bool)
    if np.any(finite_pos):
        dist_bd = np.abs(pos[finite_pos, None] - bd_centers[None, :])
        in_black_dot[finite_pos] = np.any(dist_bd < bd_radii[None, :], axis=1)

    # 2. Identify GOOD SKY fibers with functional cobras
    n_cobras = int(bench.cobras.nCobras)
    target_type = np.asarray(design.targetType, dtype=int)
    fiber_status = np.asarray(design.fiberStatus, dtype=int)
    cobra_ids_all = np.asarray(design.cobraId, dtype=int)
    fiber_ids_all = np.asarray(design.fiberId, dtype=int)

    sky_mask = (
        (target_type == int(TargetType.SKY))
        & (fiber_status == int(FiberStatus.GOOD))
        & (cobra_ids_all >= 1)
        & (cobra_ids_all <= n_cobras)
    )
    sky_idx = np.where(sky_mask)[0]
    sky_cobra_0 = cobra_ids_all[sky_idx] - 1
    good_cobra = np.asarray(bench.cobras.isGood, dtype=bool)[sky_cobra_0]
    sky_idx = sky_idx[good_cobra]
    sky_cobra_0 = sky_cobra_0[good_cobra]

    sky_centers = np.asarray(bench.cobras.centers, dtype=complex)[sky_cobra_0]
    sky_rmin = np.asarray(bench.cobras.rMin, dtype=float)[sky_cobra_0]
    sky_rmax = np.asarray(bench.cobras.rMax, dtype=float)[sky_cobra_0]
    sky_fids = fiber_ids_all[sky_idx]
    sky_cids = cobra_ids_all[sky_idx]

    # 3. Nearest currently assigned non-SKY fiber in PFI (mm)
    assigned_mask = np.isin(
        target_type,
        [int(TargetType.SCIENCE), int(TargetType.FLUXSTD), int(TargetType.ENGINEERING)],
    )
    pfi_nom = np.asarray(design.pfiNominal, dtype=float)
    assigned_valid = assigned_mask & np.all(np.isfinite(pfi_nom), axis=1)
    assigned_pos = pfi_nom[assigned_valid, 0] + 1j * pfi_nom[assigned_valid, 1]

    nearest_assigned = np.full(n_targets, np.nan, dtype=float)
    if np.any(finite_pos) and len(assigned_pos) > 0:
        dist_assigned = np.abs(pos[finite_pos, None] - assigned_pos[None, :])
        nearest_assigned[finite_pos] = np.min(dist_assigned, axis=1)

    # 4. Patrol ring check against all GOOD SKY cobras
    sky_reachable = np.zeros(n_targets, dtype=bool)
    n_sky_cobras = np.zeros(n_targets, dtype=int)
    sky_fiber_ids_out: list[str] = [""] * n_targets
    sky_cobra_ids_out: list[str] = [""] * n_targets

    if len(sky_centers) > 0 and np.any(finite_pos):
        dist_sky = np.abs(pos[:, None] - sky_centers[None, :])
        can_reach = (
            finite_pos[:, None]
            & (~in_black_dot[:, None])
            & (dist_sky >= sky_rmin[None, :])
            & (dist_sky <= sky_rmax[None, :])
        )
        for i in range(n_targets):
            hits = np.where(can_reach[i])[0]
            if len(hits) == 0:
                continue
            # Sort candidate SKY cobras by distance from cobra center
            order = hits[np.argsort(dist_sky[i, hits])]
            sky_reachable[i] = True
            n_sky_cobras[i] = len(order)
            sky_fiber_ids_out[i] = ";".join(str(int(sky_fids[k])) for k in order)
            sky_cobra_ids_out[i] = ";".join(str(int(sky_cids[k])) for k in order)

    return pd.DataFrame(
        {
            "pfi_x_mm": pfi_x,
            "pfi_y_mm": pfi_y,
            "in_black_dot": in_black_dot,
            "sky_reachable": sky_reachable,
            "n_sky_cobras": n_sky_cobras,
            "sky_fiber_ids": sky_fiber_ids_out,
            "sky_cobra_ids": sky_cobra_ids_out,
            "nearest_assigned_mm": nearest_assigned,
        }
    )


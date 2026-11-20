"""Stage 3: Reassign SKY fibers to SN candidates with collision checks and write new pfsDesign FITS."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ics.cobraOps.CollisionSimulator import CollisionSimulator
from ics.cobraOps.TargetGroup import TargetGroup
from pfs.datamodel import FiberStatus, PfsDesign, TargetType
from pfs.datamodel.utils import calculate_pfsDesignId

from .cobra_reachability import (
    DEFAULT_BLACK_DOT_MARGIN,
    get_bench,
    sky_to_pfi,
)

DEFAULT_FIDUCIAL_AVOID_MM = 2.9
DEFAULT_MIN_COBRA_SEP_MM = 2.1
DEFAULT_MIN_MAG = 18.0
DEFAULT_PROPOSAL_ID: str | None = None
DEFAULT_CAT_ID: int | None = None
DEFAULT_OB_PREFIX = "rubin_"
DEFAULT_NAME_SUFFIX = "_sn"

ALL_LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
NJY_TO_AB_ZP = 31.4


def _latest_science_mag(row: pd.Series) -> tuple[float | None, str]:
    """Return (AB mag, band) of the latest positive scienceFlux detection across all bands."""
    best_mjd = -np.inf
    best_flux = np.nan
    best_band = ""
    for b in ALL_LSST_BANDS:
        flux = row.get(f"scienceFlux_{b}", np.nan)
        mjd = row.get(f"mjd_{b}", np.nan)
        try:
            fval = float(flux)
            mval = float(mjd)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fval) and fval > 0.0 and np.isfinite(mval) and mval > best_mjd:
            best_mjd = mval
            best_flux = fval
            best_band = b
    if best_band:
        return float(-2.5 * np.log10(best_flux) + NJY_TO_AB_ZP), best_band
    for col in ("science_mag", "psf_mag"):
        val = row.get(col, np.nan)
        try:
            m = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(m):
            return m, str(row.get("rep_band", row.get("r:band", "")))
    return None, ""


def _infer_science_metadata(orig: PfsDesign) -> tuple[int, str]:
    """Infer dominant (catId, proposalId) from SCIENCE fibers in ``orig``."""
    sci = np.asarray(orig.targetType, dtype=int) == int(TargetType.SCIENCE)

    cat_id = 10099
    if np.any(sci):
        sci_cats = np.asarray(orig.catId[sci], dtype=int)
        sci_cats = sci_cats[sci_cats > 0]
        if len(sci_cats) > 0:
            vals, counts = np.unique(sci_cats, return_counts=True)
            cat_id = int(vals[np.argmax(counts)])

    proposal_id = "N/A"
    if np.any(sci):
        sci_props = [str(p).strip() for p in orig.proposalId[sci]]
        sci_props = [p for p in sci_props if p and p.upper() != "N/A"]
        if sci_props:
            vals, counts = np.unique(sci_props, return_counts=True)
            proposal_id = str(vals[np.argmax(counts)])

    return cat_id, proposal_id

HSC_FILTER_NAMES = ["g_hsc", "r2_hsc", "i2_hsc", "z_hsc", "y_hsc"]
LSST_BANDS = ["g", "r", "i", "z", "y"]

_PRIORITY_ORDER = {
    ("positive_real", "galaxy"): 0,
    ("positive_real", "no_host"): 1,
    ("negative_real", "galaxy"): 2,
    ("positive_real", "uncertain"): 3,
}


def _silence_cobra_loggers() -> None:
    for name in ("cobraCoach", "engineer", "pfi", "butler"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _design_to_target_positions(design: PfsDesign, n_cobras: int) -> np.ndarray:
    """Extract the 2394-element complex PFI position array for active cobras."""
    positions = np.full(n_cobras, TargetGroup.NULL_TARGET_POSITION, dtype=complex)
    active_types = [
        int(TargetType.SCIENCE),
        int(TargetType.SKY),
        int(TargetType.FLUXSTD),
    ]
    valid = (
        (design.cobraId >= 1)
        & (design.cobraId <= n_cobras)
        & np.all(np.isfinite(design.pfiNominal), axis=1)
        & (design.fiberStatus == int(FiberStatus.GOOD))
        & np.isin(design.targetType, active_types)
    )
    cidx = np.asarray(design.cobraId[valid], dtype=int) - 1
    positions[cidx] = design.pfiNominal[valid, 0] + 1j * design.pfiNominal[valid, 1]
    return positions


def _run_collision_sim(bench, positions: np.ndarray) -> CollisionSimulator:
    _silence_cobra_loggers()
    sim = CollisionSimulator(bench, TargetGroup(positions))
    sim.run()
    return sim


def _parse_int_list(value) -> list[int]:
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(";"):
        part = part.strip()
        if part:
            out.append(int(float(part)))
    return out


def _candidate_priority_key(row: pd.Series) -> tuple[int, float]:
    order = _PRIORITY_ORDER.get(
        (str(row.get("s1_decision", "")), str(row.get("step2_decision", ""))),
        99,
    )
    prob = float(row.get("s1_prob_positive_real", 0.0))
    if not np.isfinite(prob):
        prob = 0.0
    return (order, -prob)


def build_modified_pfs_design(
    orig: PfsDesign,
    assignments: list[dict],
    *,
    proposal_id: str | None = DEFAULT_PROPOSAL_ID,
    cat_id: int | None = DEFAULT_CAT_ID,
    ob_prefix: str = DEFAULT_OB_PREFIX,
    name_suffix: str = DEFAULT_NAME_SUFFIX,
) -> PfsDesign:
    """Create a new PfsDesign with selected SKY fibers reassigned to SN candidates."""
    inferred_cat_id, inferred_proposal_id = _infer_science_metadata(orig)
    eff_cat_id = int(cat_id) if cat_id is not None else inferred_cat_id
    eff_proposal_id = str(proposal_id) if proposal_id is not None else inferred_proposal_id
    fiberId = np.array(orig.fiberId, copy=True)
    cobraId = np.array(orig.cobraId, copy=True) if getattr(orig, "cobraId", None) is not None else None
    tract = np.array(orig.tract, copy=True)
    patch = np.array(orig.patch, copy=True)
    ra = np.array(orig.ra, copy=True)
    dec = np.array(orig.dec, copy=True)
    catId = np.array(orig.catId, copy=True)
    objId = np.array(orig.objId, copy=True)
    targetType = np.array(orig.targetType, copy=True)
    fiberStatus = np.array(orig.fiberStatus, copy=True)
    epoch = np.array(orig.epoch, copy=True)
    pmRa = np.array(orig.pmRa, copy=True)
    pmDec = np.array(orig.pmDec, copy=True)
    parallax = np.array(orig.parallax, copy=True)
    pfiNominal = np.array(orig.pfiNominal, copy=True)

    # Ensure string arrays have sufficient width for new proposalId / obCode
    proposal_list = [str(x) for x in orig.proposalId]
    obcode_list = [str(x) for x in orig.obCode]

    fiberFlux = [np.array(arr, dtype=float, copy=True) for arr in orig.fiberFlux]
    psfFlux = [np.array(arr, dtype=float, copy=True) for arr in orig.psfFlux]
    totalFlux = [np.array(arr, dtype=float, copy=True) for arr in orig.totalFlux]
    fiberFluxErr = [np.array(arr, dtype=float, copy=True) for arr in orig.fiberFluxErr]
    psfFluxErr = [np.array(arr, dtype=float, copy=True) for arr in orig.psfFluxErr]
    totalFluxErr = [np.array(arr, dtype=float, copy=True) for arr in orig.totalFluxErr]
    filterNames = [list(names) for names in orig.filterNames]

    for a in assignments:
        idx = int(a["fiber_index"])
        oid = int(a["diaObjectId"])
        ra[idx] = float(a["ra"])
        dec[idx] = float(a["dec"])
        pfiNominal[idx, 0] = float(a["pfi_x_mm"])
        pfiNominal[idx, 1] = float(a["pfi_y_mm"])
        targetType[idx] = int(TargetType.SCIENCE)
        catId[idx] = eff_cat_id
        objId[idx] = oid
        tract[idx] = 1
        patch[idx] = "1,1"
        epoch[idx] = "J2000.0"
        pmRa[idx] = 0.0
        pmDec[idx] = 0.0
        parallax[idx] = 1e-7
        proposal_list[idx] = eff_proposal_id
        obcode_list[idx] = f"{ob_prefix}{oid}"[:30]

        flux_vals = np.array([float(a.get(f"scienceFlux_{b}", np.nan)) for b in LSST_BANDS], dtype=float)
        ferr_vals = np.array([float(a.get(f"scienceFluxErr_{b}", np.nan)) for b in LSST_BANDS], dtype=float)
        nan5 = np.full(len(HSC_FILTER_NAMES), np.nan, dtype=float)

        filterNames[idx] = list(HSC_FILTER_NAMES)
        psfFlux[idx] = flux_vals
        psfFluxErr[idx] = ferr_vals
        fiberFlux[idx] = nan5.copy()
        fiberFluxErr[idx] = nan5.copy()
        totalFlux[idx] = nan5.copy()
        totalFluxErr[idx] = nan5.copy()

    proposalId_arr = np.array(proposal_list, dtype=str)
    obCode_arr = np.array(obcode_list, dtype=str)

    new_variant = int(orig.variant) + 1
    design_id0 = int(orig.pfsDesignId)
    new_design_id = int(calculate_pfsDesignId(fiberId, ra, dec, variant=new_variant))
    base_name = str(orig.designName).strip() if orig.designName else f"0x{design_id0:016x}"
    new_design_name = f"{base_name}{name_suffix}"

    return PfsDesign(
        pfsDesignId=new_design_id,
        raBoresight=float(orig.raBoresight),
        decBoresight=float(orig.decBoresight),
        posAng=float(orig.posAng),
        arms=orig.arms,
        fiberId=fiberId,
        tract=tract,
        patch=patch,
        ra=ra,
        dec=dec,
        catId=catId,
        objId=objId,
        targetType=targetType,
        fiberStatus=fiberStatus,
        epoch=epoch,
        pmRa=pmRa,
        pmDec=pmDec,
        parallax=parallax,
        proposalId=proposalId_arr,
        obCode=obCode_arr,
        fiberFlux=fiberFlux,
        psfFlux=psfFlux,
        totalFlux=totalFlux,
        fiberFluxErr=fiberFluxErr,
        psfFluxErr=psfFluxErr,
        totalFluxErr=totalFluxErr,
        filterNames=filterNames,
        pfiNominal=pfiNominal,
        guideStars=orig.guideStars,
        designName=new_design_name,
        variant=new_variant,
        designId0=design_id0,
        obstime=getattr(orig, "obstime", None),
        versions=getattr(orig, "versions", None),
        cobraId=cobraId,
    )


def reassign_design_file(
    design_path: str | Path,
    candidates_df: pd.DataFrame,
    *,
    bench=None,
    out_dir: str | Path | None = None,
    proposal_id: str | None = DEFAULT_PROPOSAL_ID,
    cat_id: int | None = DEFAULT_CAT_ID,
    ob_prefix: str = DEFAULT_OB_PREFIX,
    name_suffix: str = DEFAULT_NAME_SUFFIX,
    obs_time: str | None = None,
    black_dot_radius_margin: float = DEFAULT_BLACK_DOT_MARGIN,
    fiducial_avoid_mm: float = DEFAULT_FIDUCIAL_AVOID_MM,
    min_cobra_sep_mm: float = DEFAULT_MIN_COBRA_SEP_MM,
    min_mag: float | None = DEFAULT_MIN_MAG,
    max_per_design: int | None = None,
    allow_trajectory_collision: bool = False,
    dry_run: bool = False,
) -> tuple[PfsDesign | None, list[dict], dict]:
    """Attempt to assign SN candidates to SKY fibers in a single pfsDesign file.

    Verifies patrol ring, black-dot avoidance, fiducial avoidance, minimum cobra
    separation, and full ``ics.cobraOps.CollisionSimulator`` checks (zero new
    endpoint collisions and, unless ``allow_trajectory_collision=True``, zero new
    trajectory collisions).
    """
    path = Path(design_path)
    orig = PfsDesign.readFits(str(path))
    if bench is None:
        bench = get_bench(black_dot_radius_margin=black_dot_radius_margin)

    n_cobras = int(bench.cobras.nCobras)
    fid_pos = (
        bench.fiducials["x_mm"].to_numpy(dtype=float)
        + 1j * bench.fiducials["y_mm"].to_numpy(dtype=float)
    )
    bd_centers = np.asarray(bench.blackDots.centers, dtype=complex)
    bd_radii = np.asarray(bench.blackDots.radius, dtype=float)

    # Map fiberId -> row index in PfsDesign
    fid_to_row = {int(fid): idx for idx, fid in enumerate(orig.fiberId)}

    # Baseline positions & collision simulation
    positions = _design_to_target_positions(orig, n_cobras)
    base_sim = _run_collision_sim(bench, positions)
    cur_n_ep = int(base_sim.nEndPointCollisions)
    cur_n_traj = int(base_sim.nCollisions)

    # Sort candidates in priority order
    cands = candidates_df.copy()
    cands["_prio"] = [_candidate_priority_key(r) for _, r in cands.iterrows()]
    cands = cands.sort_values("_prio").drop(columns=["_prio"]).reset_index(drop=True)

    # Use design's own obstime if not overridden
    eff_obstime = obs_time or (str(orig.obstime) if getattr(orig, "obstime", None) else None)
    pfi_x_all, pfi_y_all = sky_to_pfi(
        cands["ra"].to_numpy(dtype=float),
        cands["dec"].to_numpy(dtype=float),
        orig,
        obs_time=eff_obstime,
    )

    used_cobra_indices: set[int] = set()
    accepted: list[dict] = []
    log_rows: list[dict] = []

    for i, row in cands.iterrows():
        oid = int(row["diaObjectId"])
        latest_mag, latest_band = _latest_science_mag(row)

        if min_mag is not None and latest_mag is not None and latest_mag <= float(min_mag):
            log_rows.append(
                {
                    "design_file": path.name,
                    "design_name": orig.designName,
                    "diaObjectId": oid,
                    "status": "rejected",
                    "assigned_fiber_id": np.nan,
                    "assigned_cobra_id": np.nan,
                    "latest_band": latest_band,
                    "latest_mag": round(latest_mag, 3),
                    "reason": f"too_bright ({latest_band}={latest_mag:.2f}mag <= {float(min_mag):.2f}mag)",
                }
            )
            continue

        if max_per_design is not None and len(accepted) >= max_per_design:
            log_rows.append(
                {
                    "design_file": path.name,
                    "design_name": orig.designName,
                    "diaObjectId": oid,
                    "status": "skipped_max_per_design",
                    "assigned_fiber_id": np.nan,
                    "assigned_cobra_id": np.nan,
                    "reason": f"Reached max_per_design={max_per_design}",
                }
            )
            continue

        px = float(pfi_x_all[i])
        py = float(pfi_y_all[i])
        new_pos = complex(px, py)

        # Black-dot and fiducial checks (independent of which SKY cobra is chosen)
        if not (np.isfinite(px) and np.isfinite(py)):
            log_rows.append(
                {
                    "design_file": path.name,
                    "design_name": orig.designName,
                    "diaObjectId": oid,
                    "status": "rejected",
                    "assigned_fiber_id": np.nan,
                    "assigned_cobra_id": np.nan,
                    "reason": "non_finite_pfi",
                }
            )
            continue

        if np.any(np.abs(new_pos - bd_centers) < bd_radii):
            log_rows.append(
                {
                    "design_file": path.name,
                    "design_name": orig.designName,
                    "diaObjectId": oid,
                    "status": "rejected",
                    "assigned_fiber_id": np.nan,
                    "assigned_cobra_id": np.nan,
                    "reason": "black_dot",
                }
            )
            continue

        min_fid_dist = float(np.min(np.abs(new_pos - fid_pos)))
        if min_fid_dist < fiducial_avoid_mm:
            log_rows.append(
                {
                    "design_file": path.name,
                    "design_name": orig.designName,
                    "diaObjectId": oid,
                    "status": "rejected",
                    "assigned_fiber_id": np.nan,
                    "assigned_cobra_id": np.nan,
                    "reason": f"fiducial_too_close ({min_fid_dist:.2f}mm < {fiducial_avoid_mm}mm)",
                }
            )
            continue

        # Candidate SKY fibers for this target
        cand_fids = _parse_int_list(row.get("sky_fiber_ids", ""))
        if not cand_fids:
            # Fallback: find all GOOD SKY fibers whose patrol ring covers new_pos
            sky_rows = np.where(
                (orig.targetType == int(TargetType.SKY))
                & (orig.fiberStatus == int(FiberStatus.GOOD))
                & (orig.cobraId >= 1)
                & (orig.cobraId <= n_cobras)
            )[0]
            for ridx in sky_rows:
                c0 = int(orig.cobraId[ridx]) - 1
                if not bench.cobras.isGood[c0]:
                    continue
                d_c = abs(new_pos - bench.cobras.centers[c0])
                if bench.cobras.rMin[c0] <= d_c <= bench.cobras.rMax[c0]:
                    cand_fids.append(int(orig.fiberId[ridx]))

        assigned_this = False
        reject_reasons: list[str] = []

        for fid in cand_fids:
            fidx = fid_to_row.get(fid)
            if fidx is None:
                continue
            if int(orig.targetType[fidx]) != int(TargetType.SKY) or int(orig.fiberStatus[fidx]) != int(FiberStatus.GOOD):
                continue
            cid = int(orig.cobraId[fidx])
            c0 = cid - 1
            if c0 in used_cobra_indices or not bench.cobras.isGood[c0]:
                reject_reasons.append(f"cobra{cid}:already_used")
                continue

            # Patrol ring check
            dist_center = float(abs(new_pos - bench.cobras.centers[c0]))
            if not (bench.cobras.rMin[c0] <= dist_center <= bench.cobras.rMax[c0]):
                reject_reasons.append(f"cobra{cid}:out_of_patrol")
                continue

            # Minimum endpoint separation check against all other active cobras
            other_mask = np.isfinite(positions.real) & np.isfinite(positions.imag)
            other_mask[c0] = False
            min_sep = float(np.min(np.abs(new_pos - positions[other_mask]))) if np.any(other_mask) else 999.0
            if min_sep < min_cobra_sep_mm:
                reject_reasons.append(f"cobra{cid}:endpoint_sep={min_sep:.2f}mm")
                continue

            # Exact CollisionSimulator verification
            trial_positions = positions.copy()
            trial_positions[c0] = new_pos
            trial_sim = _run_collision_sim(bench, trial_positions)

            if trial_sim.endPointCollisions[c0] or int(trial_sim.nEndPointCollisions) > cur_n_ep:
                reject_reasons.append(
                    f"cobra{cid}:endpoint_collision(dEP={int(trial_sim.nEndPointCollisions) - cur_n_ep})"
                )
                continue

            if not allow_trajectory_collision:
                if trial_sim.collisions[c0] or int(trial_sim.nCollisions) > cur_n_traj:
                    reject_reasons.append(
                        f"cobra{cid}:trajectory_collision(dTraj={int(trial_sim.nCollisions) - cur_n_traj})"
                    )
                    continue

            # Accepted! Commit cobra move
            positions = trial_positions
            cur_n_ep = int(trial_sim.nEndPointCollisions)
            cur_n_traj = int(trial_sim.nCollisions)
            used_cobra_indices.add(c0)

            rec = {
                "fiber_index": fidx,
                "fiber_id": fid,
                "cobra_id": cid,
                "diaObjectId": oid,
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
                "pfi_x_mm": px,
                "pfi_y_mm": py,
                "min_neighbor_sep_mm": min_sep,
                "min_fiducial_dist_mm": min_fid_dist,
                "s1_decision": str(row.get("s1_decision", "")),
                "step2_decision": str(row.get("step2_decision", "")),
                "decision_detail": str(row.get("decision_detail", "")),
            }
            for b in LSST_BANDS:
                rec[f"scienceFlux_{b}"] = row.get(f"scienceFlux_{b}", np.nan)
                rec[f"scienceFluxErr_{b}"] = row.get(f"scienceFluxErr_{b}", np.nan)
            accepted.append(rec)

            log_rows.append(
                {
                    "design_file": path.name,
                    "design_name": orig.designName,
                    "diaObjectId": oid,
                    "status": "assigned",
                    "assigned_fiber_id": fid,
                    "assigned_cobra_id": cid,
                    "latest_band": latest_band,
                    "latest_mag": round(latest_mag, 3) if latest_mag is not None else np.nan,
                    "min_neighbor_sep_mm": round(min_sep, 3),
                    "min_fiducial_dist_mm": round(min_fid_dist, 3),
                    "reason": "ok",
                }
            )
            assigned_this = True
            break

        if not assigned_this:
            log_rows.append(
                {
                    "design_file": path.name,
                    "design_name": orig.designName,
                    "diaObjectId": oid,
                    "status": "rejected",
                    "assigned_fiber_id": np.nan,
                    "assigned_cobra_id": np.nan,
                    "min_neighbor_sep_mm": np.nan,
                    "min_fiducial_dist_mm": round(min_fid_dist, 3),
                    "reason": "; ".join(reject_reasons) if reject_reasons else "no_available_sky_cobra",
                }
            )

    new_design: PfsDesign | None = None
    out_file = ""
    if accepted:
        new_design = build_modified_pfs_design(
            orig,
            accepted,
            proposal_id=proposal_id,
            cat_id=cat_id,
            ob_prefix=ob_prefix,
            name_suffix=name_suffix,
        )
        if out_dir is not None and not dry_run:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            new_design.write(dirName=str(out_path))
            out_file = f"pfsDesign-0x{new_design.pfsDesignId:016x}.fits"

    summary = {
        "orig_design_file": path.name,
        "orig_design_name": orig.designName,
        "orig_design_id": f"0x{orig.pfsDesignId:016x}",
        "n_candidates": int(len(cands)),
        "n_assigned": int(len(accepted)),
        "n_rejected": int(len(cands) - len(accepted)),
        "new_design_file": out_file,
        "new_design_name": new_design.designName if new_design is not None else "",
        "new_design_id": f"0x{new_design.pfsDesignId:016x}" if new_design is not None else "",
        "assigned_fiber_ids": ";".join(str(a["fiber_id"]) for a in accepted),
        "assigned_obj_ids": ";".join(str(a["diaObjectId"]) for a in accepted),
    }
    return new_design, log_rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reassign-pfs-design",
        description=(
            "Reassign SKY fibers in pfsDesign FITS files to SN candidates from "
            "design_sn_candidates.csv, verifying cobra patrol rings, black dots, "
            "fiducial avoidance, and CollisionSimulator checks."
        ),
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to design_sn_candidates.csv generated by select-sn-targets.",
    )
    parser.add_argument(
        "--design-dir",
        required=True,
        help="Directory containing the original pfsDesign-*.fits files.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default="modified_designs",
        help="Output directory for new pfsDesign-*.fits files and reports (default: modified_designs).",
    )
    parser.add_argument(
        "--proposal-id",
        default=DEFAULT_PROPOSAL_ID,
        help="Override proposalId for reassigned SN targets (default: inherit from original SCIENCE fibers).",
    )
    parser.add_argument(
        "--cat-id",
        type=int,
        default=DEFAULT_CAT_ID,
        help="Override catId for reassigned SN targets (default: inherit from original SCIENCE fibers).",
    )
    parser.add_argument(
        "--ob-prefix",
        default=DEFAULT_OB_PREFIX,
        help=f"Prefix for obCode (obCode = <prefix><diaObjectId>, default: {DEFAULT_OB_PREFIX}).",
    )
    parser.add_argument(
        "--name-suffix",
        default=DEFAULT_NAME_SUFFIX,
        help=f"Suffix appended to designName (default: {DEFAULT_NAME_SUFFIX}).",
    )
    parser.add_argument(
        "--obstime",
        default=None,
        help="Override observation time (UTC ISO) for sky->PFI transform.",
    )
    parser.add_argument(
        "--black-dot-margin",
        type=float,
        default=DEFAULT_BLACK_DOT_MARGIN,
        help=f"Black-dot avoidance margin (default: {DEFAULT_BLACK_DOT_MARGIN}).",
    )
    parser.add_argument(
        "--fiducial-avoid-mm",
        type=float,
        default=DEFAULT_FIDUCIAL_AVOID_MM,
        help=f"Minimum distance from fiducials in mm (default: {DEFAULT_FIDUCIAL_AVOID_MM}).",
    )
    parser.add_argument(
        "--min-cobra-sep-mm",
        type=float,
        default=DEFAULT_MIN_COBRA_SEP_MM,
        help=f"Minimum endpoint separation from other cobras in mm (default: {DEFAULT_MIN_COBRA_SEP_MM}).",
    )
    parser.add_argument(
        "--min-mag",
        type=float,
        default=DEFAULT_MIN_MAG,
        help=(
            f"Bright magnitude limit (AB mag, any band): transients whose latest detection "
            f"is brighter than or equal to this value (mag <= --min-mag) are not assigned "
            f"(default: {DEFAULT_MIN_MAG})."
        ),
    )
    parser.add_argument(
        "--max-per-design",
        type=int,
        default=None,
        help="Optional maximum number of SKY fibers to reassign per pfsDesign.",
    )
    parser.add_argument(
        "--allow-trajectory-collision",
        action="store_true",
        help="Only enforce endpoint collision avoidance (allow trajectory collisions).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all collision checks and write CSV reports without writing FITS files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cand_path = Path(args.candidates)
    design_dir = Path(args.design_dir)
    out_dir = Path(args.out_dir)

    if not cand_path.is_file():
        print(f"Error: Candidates CSV not found: {cand_path}", file=sys.stderr)
        return 1
    if not design_dir.is_dir():
        print(f"Error: Design directory not found: {design_dir}", file=sys.stderr)
        return 1

    df = pd.read_csv(cand_path)
    if df.empty:
        print("No candidates in CSV; nothing to reassign.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    bench = get_bench(black_dot_radius_margin=args.black_dot_margin)

    groups = list(df.groupby("design_file", sort=True))
    print(f"Processing {len(groups)} pfsDesign file(s) with {len(df)} candidate-design pairs...")

    summaries: list[dict] = []
    all_logs: list[dict] = []

    for i, (design_file, gdf) in enumerate(groups, start=1):
        dpath = design_dir / str(design_file)
        if not dpath.is_file():
            print(f"[{i}/{len(groups)}] MISSING: {dpath}", file=sys.stderr)
            continue

        _, logs, summary = reassign_design_file(
            dpath,
            gdf,
            bench=bench,
            out_dir=out_dir,
            proposal_id=args.proposal_id,
            cat_id=args.cat_id,
            ob_prefix=args.ob_prefix,
            name_suffix=args.name_suffix,
            obs_time=args.obstime,
            black_dot_radius_margin=args.black_dot_margin,
            fiducial_avoid_mm=args.fiducial_avoid_mm,
            min_cobra_sep_mm=args.min_cobra_sep_mm,
            min_mag=args.min_mag,
            max_per_design=args.max_per_design,
            allow_trajectory_collision=args.allow_trajectory_collision,
            dry_run=args.dry_run,
        )
        summaries.append(summary)
        all_logs.extend(logs)

        out_label = summary["new_design_file"] or "(no change)"
        print(
            f"[{i}/{len(groups)}] {summary['orig_design_name']} ({design_file}): "
            f"assigned {summary['n_assigned']}/{summary['n_candidates']} -> {out_label}"
        )

    sum_df = pd.DataFrame(summaries)
    log_df = pd.DataFrame(all_logs)
    sum_df.to_csv(out_dir / "reassignment_summary.csv", index=False)
    log_df.to_csv(out_dir / "reassignment_log.csv", index=False)

    tot_cand = int(sum_df["n_candidates"].sum()) if not sum_df.empty else 0
    tot_assg = int(sum_df["n_assigned"].sum()) if not sum_df.empty else 0
    n_mod = int((sum_df["n_assigned"] > 0).sum()) if not sum_df.empty else 0
    print(
        f"\nDone: assigned {tot_assg}/{tot_cand} candidate-design pairs across "
        f"{n_mod}/{len(summaries)} designs -> {out_dir}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Stage 2: Supernova candidate classification using RubinTargetSelection (stamp_classifier).

Runs the frozen ``snp_v1`` bundle (Step 1 Difference SimCLR+RF -> Visit check ->
Object aggregation & moving flag -> Step 2 Host morphology Group A + Group B RF)
directly on Fink/LSST API data without requiring a PostgreSQL database.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from astropy.time import Time

from .fink_client import FINK_LSST_API_URL, ensure_cutouts_cached

DEFAULT_BUNDLE_DIR = "/mnt/ugnas/work/LSST/fink2/models/sn_pipeline/snp_v1"
DEFAULT_LOCAL_CUTOUTS = "/mnt/ugnas/work/LSST/fink2/cutouts"
DEFAULT_EXCLUDED_VISITS = "/mnt/ugnas/work/LSST/fink2/data/excluded_visits.csv"
DEFAULT_CUTOUT_CACHE = "cache/cutouts"


def _md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bundle_dir(bundle_dir: str | Path) -> dict:
    """Load and verify a frozen sn_pipeline model bundle from an explicit directory."""
    d = Path(bundle_dir)
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        m = json.load(f)
    for name, expected_md5 in m["md5"].items():
        fpath = d / name
        if not fpath.exists():
            raise FileNotFoundError(f"Missing bundle file: {fpath}")
        if _md5(fpath) != expected_md5:
            raise RuntimeError(f"{name} in bundle '{d}' failed MD5 verification.")
    with open(d / "host_thresholds.json") as f:
        m["host_thresholds"] = json.load(f)
    m["dir"] = str(d)
    return m


def _stretch_stamp(img: np.ndarray, *, symmetric: bool = False) -> np.ndarray:
    """Arcsinh stretch a 2D cutout to [0, 255] uint8 for inspection plots."""
    arr = np.nan_to_num(np.asarray(img, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if symmetric:
        scale = float(np.percentile(np.abs(arr), 99.0))
        if scale <= 0:
            return np.full(arr.shape, 128, dtype=np.uint8)
        norm = np.arcsinh(arr / (0.2 * scale))
        vmax = float(np.max(np.abs(norm))) or 1.0
        scaled = 0.5 + 0.5 * (norm / vmax)
    else:
        lo, hi = np.percentile(arr, [5.0, 99.5])
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        norm = np.arcsinh((arr - lo) / max(0.1 * (hi - lo), 1e-6))
        nmin, nmax = float(np.min(norm)), float(np.max(norm))
        scaled = (norm - nmin) / max(nmax - nmin, 1e-6)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _write_candidates_pdf(
    path: str | Path,
    summary: pd.DataFrame,
    fits_root: str,
    title: str,
    rows: int = 8,
    cols: int = 5,
) -> None:
    """Generate a self-contained Sci | Tmp | Dif triplet PDF for SN candidates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from matplotlib.backends.backend_pdf import PdfPages
    from stamp_classifier.dataset import get_cutout_path

    per = rows * cols
    n_pages = (len(summary) + per - 1) // per
    with PdfPages(str(path)) as pdf:
        for page in range(n_pages):
            chunk = summary.iloc[page * per : (page + 1) * per]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.0), squeeze=False)
            fig.suptitle(
                f"{title} | Page {page + 1}/{n_pages}\nLayout: Sci | Tmp | Dif",
                fontsize=12,
                fontweight="bold",
            )
            flat = axes.flatten()
            for i, r in enumerate(chunk.itertuples()):
                ax = flat[i]
                panels = []
                for itype, sym in (("science", False), ("template", False), ("difference", True)):
                    p = get_cutout_path(
                        fits_root, int(r.diaObjectId), float(r.rep_mjd), str(r.rep_band), int(r.rep_diaSourceId), itype
                    )
                    if os.path.exists(p):
                        data = fits.getdata(p)
                        panels.append(_stretch_stamp(data, symmetric=sym))
                    else:
                        panels.append(np.zeros((30, 30), dtype=np.uint8))
                sep = np.full((panels[0].shape[0], 2), 255, dtype=np.uint8)
                triplet_img = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
                ax.imshow(triplet_img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                date = Time(r.rep_mjd, format="mjd", scale="tai").utc.iso[5:10]
                s1_dec = getattr(r, "s1_decision", "positive_real")
                s1_p = r.s1_prob_negative_real if s1_dec == "negative_real" else r.s1_prob_positive_real
                ax.set_title(
                    f"{r.diaObjectId} | {r.rep_band} | {date} | {r.decision_detail}\n"
                    f"{s1_dec}:{s1_p:.2f} bogus:{r.s1_prob_bogus:.2f} (N={r.n_nights_total}n)",
                    fontsize=7,
                )
                ax.axis("off")
            for j in range(len(chunk), len(flat)):
                flat[j].axis("off")
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, dpi=120)
            plt.close(fig)


def _load_excluded_visit_set(path: str | Path | None) -> set[int]:
    if not path or not os.path.exists(str(path)):
        return set()
    df = pd.read_csv(path)
    if "visit" not in df.columns or df.empty:
        return set()
    return set(pd.to_numeric(df["visit"], errors="coerce").dropna().astype(np.int64))


def classify_sources(
    all_sources_df: pd.DataFrame,
    *,
    start: str,
    end: str | None,
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
    out_dir: str | Path = "sn_output",
    cache_root: str | Path = DEFAULT_CUTOUT_CACHE,
    extra_fits_roots: Sequence[str | Path] = (DEFAULT_LOCAL_CUTOUTS,),
    excluded_visits_path: str | Path | None = DEFAULT_EXCLUDED_VISITS,
    min_nights: int = 2,
    min_nights_no_host: int = 2,
    include_uncertain: bool = True,
    include_negative_galaxy: bool = False,
    keep_suspicious_visits: bool = False,
    per_night_band: str = "snr",
    download_workers: int = 8,
    encode_workers: int = 4,
    make_pdf: bool = True,
    fink_base_url: str = FINK_LSST_API_URL,
) -> tuple[pd.DataFrame, dict]:
    """Run the frozen ``stamp_classifier`` SN pipeline on API-fetched sources.

    Parameters
    ----------
    all_sources_df : pd.DataFrame
        Full detection history of candidate ``diaObjectId``s returned by
        ``FinkLSSTClient.get_sources_for_classifier()``.
    start, end : str
        UTC date range (ISO ``YYYY-MM-DD`` or timestamp). If ``end`` is ``None``,
        defaults to current UTC time.
    min_nights : int
        Minimum number of distinct observing nights (``n_nights_total >= min_nights``).
        Applied *before* downloading cutouts to save API calls. Default: 2.
    """
    from stamp_classifier.group_b import (
        CLASSES as B_CLASSES,
        FINAL_DECISION as B_FINAL,
        HOST_A_TO_DECISION,
    )
    from stamp_classifier.host_morphology import (
        HOST_MORPH_FEATURE_NAMES,
        clean_for_rf,
        host_features_batch,
        object_host_morph,
    )
    from stamp_classifier.model_registry import decide
    from stamp_classifier.morphology import extract_morphological_features_batch
    from stamp_classifier.object_aggregation import (
        add_detection_history,
        aggregate_objects,
        flag_moving,
        select_step2_sources,
        summarize_detections,
    )
    from stamp_classifier.sn_pipeline import (
        BAND_ORDER,
        PER_NIGHT_BAND,
        S1_CLASSES,
        candidates_pdf,
        check_visits,
        encode,
    )

    t0 = time.time()
    B = load_bundle_dir(bundle_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_root)
    cache_path.mkdir(parents=True, exist_ok=True)

    t_start = Time(start)
    t_end = Time(end) if end else Time.now()
    d0 = int(np.floor(t_start.mjd))
    d1 = int(np.floor(t_end.mjd))

    info: dict[str, object] = {
        "start": start,
        "end": end or t_end.utc.iso[:10],
        "dayObs_range": [d0, d1],
        "bundle_dir": str(bundle_dir),
        "bundle_version": B.get("version", Path(bundle_dir).name),
        "cache_root": str(cache_path),
        "min_nights": min_nights,
        "min_nights_no_host": min_nights_no_host,
        "per_night_band": per_night_band,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if all_sources_df.empty:
        print("[1] No sources provided to classifier.")
        return pd.DataFrame(), info

    # 1. Full detection history from API sources (replaces DB diaSource + prvDiaSources)
    hist = summarize_detections(
        all_sources_df[["diaObjectId", "diaSourceId", "midpointMjdTai", "band", "ra", "dec"]]
    )
    n_obj_total = len(hist)
    effective_min_nights = max(1, int(min_nights))
    eligible_ids = set(
        hist.loc[hist["n_nights_total"] >= effective_min_nights, "diaObjectId"].astype(np.int64)
    )
    print(
        f"[1] Detection history: {n_obj_total} objects ({len(all_sources_df)} total alerts) -> "
        f"{len(eligible_ids)} objects with n_nights_total >= {effective_min_nights}"
    )
    info["n_objects_input"] = int(n_obj_total)
    info["n_objects_min_nights"] = int(len(eligible_ids))

    if not eligible_ids:
        return pd.DataFrame(), info

    # Filter detections to [d0, d1] dayObs window for the eligible objects
    day_obs = np.floor(all_sources_df["midpointMjdTai"].to_numpy(dtype=float) - 0.5).astype(int)
    in_window = (
        all_sources_df["diaObjectId"].isin(eligible_ids)
        & (day_obs >= d0)
        & (day_obs <= d1)
    )
    raw = all_sources_df.loc[in_window].copy().reset_index(drop=True)
    if raw.empty:
        print(f"    0 detections within dayObs {d0}..{d1} for eligible objects.")
        return pd.DataFrame(), info

    src = PER_NIGHT_BAND[per_night_band](raw)
    print(
        f"    {len(raw)} detections in dayObs {d0}..{d1} -> "
        f"{len(src)} sources ({src['diaObjectId'].nunique()} objects) after "
        f"per-night-band ({per_night_band}) deduplication"
    )

    # 2. Ensure cutouts are cached locally (reuse local fink2 cutouts or download via Fink API)
    cutout_stats = ensure_cutouts_cached(
        src,
        cache_path,
        extra_fits_roots=extra_fits_roots,
        workers=download_workers,
        base_url=fink_base_url,
    )
    print(
        f"[2] Cutouts ready: cached={cutout_stats['already_cached']}, "
        f"linked_local={cutout_stats['linked_local']}, "
        f"downloaded={cutout_stats['downloaded']}, failed={cutout_stats['failed']}"
    )
    info["cutout_stats"] = cutout_stats

    fits_root_str = str(cache_path)

    # 3. Step 1: Difference SimCLR encoder + morphological features + RF
    emb1, img1 = encode(
        src,
        ["difference"],
        os.path.join(B["dir"], "step1_encoder.pt"),
        fits_root_str,
        workers=encode_workers,
    )
    morph1 = extract_morphological_features_batch(img1)
    has_cut = np.abs(img1).reshape(len(img1), -1).sum(1) > 0
    rf1 = joblib.load(os.path.join(B["dir"], "step1_rf.joblib"))
    raw_p = rf1.predict_proba(np.hstack([emb1, morph1]))
    P1 = np.zeros((len(src), 3))
    for j, k in enumerate(rf1.classes_):
        P1[:, int(k)] = raw_p[:, j]
    src["s1_prob_positive_real"] = P1[:, 0]
    src["s1_prob_negative_real"] = P1[:, 1]
    src["s1_prob_bogus"] = P1[:, 2]
    src["s1_decision"] = decide(P1, S1_CLASSES, B["step1"]["thresholds"])
    src.loc[~has_cut, "s1_decision"] = "no_cutout"
    print(f"[3] Step 1: {src['s1_decision'].value_counts().to_dict()}")

    # 4. Visit quality check
    vchk = check_visits(src, fits_root_str, B["visit_check"])
    vchk.to_csv(out_path / "visit_check.csv", index=False)
    known_bad = _load_excluded_visit_set(excluded_visits_path)
    sus = set(vchk.loc[vchk["suspicious"], "visit"].astype(np.int64)) if len(vchk) else set()
    bad = known_bad | (set() if keep_suspicious_visits else sus)
    src["bad_visit"] = src["visit"].isin(bad)
    src.loc[src["bad_visit"], "s1_decision"] = "bogus"
    info["suspicious_visits"] = sorted(int(v) for v in sus)
    info["excluded_visits_applied"] = sorted(int(v) for v in bad & set(src["visit"].astype(np.int64)))

    # 5. Object aggregation + detection history + moving flag
    agg_in = src.rename(
        columns={
            "s1_decision": "decision",
            "s1_prob_positive_real": "prob_positive_real",
            "s1_prob_negative_real": "prob_negative_real",
            "s1_prob_bogus": "prob_bogus",
        }
    )
    agg_in = agg_in[agg_in["decision"] != "no_cutout"]
    if agg_in.empty:
        print("[4] No sources with valid cutouts after Step 1.")
        return pd.DataFrame(), info

    obj = aggregate_objects(agg_in)
    obj = flag_moving(
        add_detection_history(obj, hist),
        B["moving"]["min_rate_arcsec_per_min"],
        B["moving"]["min_sep_arcsec"],
    )
    s2 = select_step2_sources(
        agg_in, obj, ["real", "uncertain"], mode="real_uncertain", include_moving=False
    )
    s2 = s2.merge(src[["diaSourceId", "ra", "dec"]], on="diaSourceId", how="left")
    print(
        f"[4] Objects: {obj['object_class'].value_counts().to_dict()}, "
        f"moving={int(obj['moving_candidate'].sum())}; "
        f"Step 2 input: {len(s2)} sources of {s2['diaObjectId'].nunique()} objects"
    )

    # 6. Step 2: Host morphology (Group A) + 4-class RF (Group B)
    F = s2[["diaSourceId", "diaObjectId", "midpointMjdTai", "band"]].copy()
    F["decision"] = pd.Series(dtype=object)
    F["decision_detail"] = pd.Series(dtype=object)
    if len(F):
        H = host_features_batch(F.to_dict("records"), fits_root_str)
        Hdf = pd.DataFrame(H, columns=HOST_MORPH_FEATURE_NAMES)
        F = pd.concat([F.reset_index(drop=True), Hdf], axis=1)
        O = object_host_morph(F, B["host_thresholds"])
        F = F.merge(
            O[["diaObjectId", "host_score", "host_morph_obj", "band_scores"]],
            on="diaObjectId",
            how="left",
        )
        F["host_group"] = np.where(
            F["host_morph_obj"].isin(["point", "extended", "uncertain"]), "A", "B"
        )
        F["decision"] = "uncertain"
        F["decision_detail"] = ""
        a = F["host_group"] == "A"
        F.loc[a, "decision"] = F.loc[a, "host_morph_obj"].map(HOST_A_TO_DECISION)
        F.loc[a, "decision_detail"] = "A:" + F.loc[a, "host_morph_obj"]

        bi = np.where(~a)[0]
        F["class_b"] = None
        if len(bi):
            Fb = F.iloc[bi]
            emb2, _ = encode(
                Fb.merge(s2[["diaSourceId", "ra", "dec"]], on="diaSourceId", how="left"),
                ["science", "template"],
                os.path.join(B["dir"], "step2_encoder.pt"),
                fits_root_str,
                workers=encode_workers,
            )
            pos = (
                pd.Series(np.arange(len(src)), index=src["diaSourceId"].values)
                .loc[Fb["diaSourceId"].values]
                .values
            )
            Xb = np.hstack(
                [
                    emb2,
                    emb1[pos],
                    clean_for_rf(morph1[pos]),
                    clean_for_rf(Fb[HOST_MORPH_FEATURE_NAMES].values),
                ]
            )
            rfb = joblib.load(os.path.join(B["dir"], "step2b_rf.joblib"))
            raw_b = rfb.predict_proba(Xb.astype(np.float32))
            Pb = np.zeros((len(Xb), len(B_CLASSES)))
            for j, k in enumerate(rfb.classes_):
                Pb[:, int(k)] = raw_b[:, j]
            cls = np.array(B_CLASSES)[Pb.argmax(1)]
            F.loc[F.index[bi], "class_b"] = cls
            F.loc[F.index[bi], "prob_b"] = Pb.max(1)
            F.loc[F.index[bi], "decision"] = [B_FINAL[c] for c in cls]
            F.loc[F.index[bi], "decision_detail"] = ["B:" + c for c in cls]
        print(f"[5] Step 2: {F['decision_detail'].value_counts().to_dict()}")

    # 7. Candidate selection
    s2_classes = ["galaxy", "no_host"] + (["uncertain"] if include_uncertain else [])
    full = src.merge(
        F.drop(columns=["midpointMjdTai", "band", "diaObjectId"], errors="ignore"),
        on="diaSourceId",
        how="left",
    )
    full = full.merge(
        obj[
            [
                "diaObjectId",
                "object_class",
                "n_nights_total",
                "n_det_total",
                "max_det_per_night",
                "first_det_mjd",
                "last_det_mjd",
                "moving_candidate",
            ]
        ],
        on="diaObjectId",
        how="left",
    )
    full.to_csv(out_path / "all_sources.csv", index=False)

    is_pos = (full["s1_decision"] == "positive_real") & full["decision"].isin(s2_classes)
    is_neg_gal = (
        (full["s1_decision"] == "negative_real") & (full["decision"] == "galaxy")
        if include_negative_galaxy
        else False
    )
    hit = full[is_pos | is_neg_gal].copy()
    if effective_min_nights > 1:
        hit = hit[hit["n_nights_total"] >= effective_min_nights]
    if min_nights_no_host is not None and min_nights_no_host > 1:
        hit = hit[~((hit["decision"] == "no_host") & (hit["n_nights_total"] < min_nights_no_host))]

    cand_ids = hit["diaObjectId"].unique()
    print(f"[6] SN candidates: {len(cand_ids)} objects")

    if len(cand_ids) == 0:
        summary = pd.DataFrame()
        summary.to_csv(out_path / "candidates.csv", index=False)
        info.update(
            {
                "n_sources": int(len(src)),
                "n_candidates": 0,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_sec": round(time.time() - t0, 1),
            }
        )
        with open(out_path / "run_info.json", "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False, default=str)
        return summary, info

    # Representative detection per object (same priority order as sn_pipeline.py)
    order_map = {
        ("positive_real", "galaxy"): 0,
        ("positive_real", "no_host"): 1,
        ("negative_real", "galaxy"): 2,
        ("positive_real", "uncertain"): 3,
    }
    combo = list(zip(hit["s1_decision"], hit["decision"]))
    hit = hit.assign(
        _order=[order_map.get(c, 99) for c in combo],
        _s1_prob=np.where(
            hit["s1_decision"] == "negative_real",
            hit["s1_prob_negative_real"],
            hit["s1_prob_positive_real"],
        ),
    )
    rep = (
        hit.sort_values(["_order", "midpointMjdTai"], ascending=[True, False])
        .groupby("diaObjectId")
        .head(1)
    )
    rep = rep.sort_values(["_order", "_s1_prob"], ascending=[True, False]).set_index("diaObjectId")

    rows = []
    det_in = raw[raw["diaObjectId"].isin(cand_ids)].sort_values("midpointMjdTai")
    for oid, r in rep.iterrows():
        d = det_in[det_in["diaObjectId"] == oid]
        latest = d.groupby("band").tail(1)
        row = {
            "diaObjectId": int(oid),
            "ra": float(d["ra"].mean()),
            "dec": float(d["dec"].mean()),
            "step2_decision": r["decision"],
            "decision_detail": r["decision_detail"],
            "host_score": r.get("host_score"),
            "band_scores": r.get("band_scores"),
            "prob_b": r.get("prob_b"),
            "s1_decision": r.get("s1_decision"),
            "s1_prob_positive_real": r["s1_prob_positive_real"],
            "s1_prob_negative_real": r.get("s1_prob_negative_real"),
            "s1_prob_bogus": r["s1_prob_bogus"],
            "rep_diaSourceId": int(r["diaSourceId"]),
            "rep_band": r["band"],
            "rep_mjd": float(r["midpointMjdTai"]),
            "n_det_in_range": int(len(d)),
            "n_nights_total": int(r.get("n_nights_total", 0)),
            "max_det_per_night": int(r.get("max_det_per_night", 0)),
            "first_det_mjd": float(r.get("first_det_mjd", np.nan)),
            "last_det_mjd": float(r.get("last_det_mjd", np.nan)),
        }
        for b in BAND_ORDER:
            x = latest[latest["band"] == b]
            row[f"scienceFlux_{b}"] = float(x["scienceFlux"].iloc[0]) if len(x) else np.nan
            row[f"scienceFluxErr_{b}"] = float(x["scienceFluxErr"].iloc[0]) if len(x) else np.nan
            row[f"mjd_{b}"] = float(x["midpointMjdTai"].iloc[0]) if len(x) else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path / "candidates.csv", index=False)

    if make_pdf and len(summary):
        pdf_path = str(out_path / "candidates.pdf")
        title = (
            f"PFS SKY-reachable SN candidates  {start} .. {info['end']}  "
            f"(bundle {info['bundle_version']}, nights>={effective_min_nights})"
        )
        _write_candidates_pdf(pdf_path, summary, fits_root_str, title=title)

    info.update(
        {
            "n_sources": int(len(src)),
            "n_candidates": int(len(summary)),
            "candidate_decisions": summary["decision_detail"].value_counts().to_dict(),
            "include_uncertain": include_uncertain,
            "include_negative_galaxy": include_negative_galaxy,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - t0, 1),
        }
    )
    with open(out_path / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)

    return summary, info

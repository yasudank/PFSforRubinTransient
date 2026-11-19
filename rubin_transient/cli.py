"""Command-line interfaces for Stage 1 (transient search) and Stage 2 (SKY-reachable SN selection)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .cobra_reachability import (
    DEFAULT_BLACK_DOT_MARGIN,
    check_sky_fiber_reachability,
    get_bench,
)
from .fink_client import DEFAULT_CONESEARCH_COLUMNS, FinkAPIError, FinkLSSTClient
from .pfs_design import (
    DEFAULT_FOV_MARGIN_ARCSEC,
    PfsDesignInfo,
    discover_pfs_designs,
    load_pfs_design,
)
from .sn_classifier import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_CUTOUT_CACHE,
    DEFAULT_EXCLUDED_VISITS,
    DEFAULT_LOCAL_CUTOUTS,
    classify_sources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find_transients",
        description=(
            "Search for Rubin/LSST transients inside the field of view of one or more "
            "pfsDesign FITS files using the Fink/LSST object API."
        ),
    )
    parser.add_argument(
        "designs",
        nargs="+",
        help="Path(s) to pfsDesign-*.fits file(s) or directory containing them.",
    )
    parser.add_argument(
        "--start",
        "--startdate",
        dest="startdate",
        required=True,
        help="Start date in UTC (e.g. '2026-11-01', '2026-11-01 00:00:00', or MJD/JD).",
    )
    parser.add_argument(
        "--stop",
        "--stopdate",
        dest="stopdate",
        default=None,
        help=(
            "Stop date in UTC (e.g. '2026-11-19', '2026-11-19 23:59:59', or MJD/JD). "
            "Date-only strings are treated as inclusive end of day (23:59:59 UTC). "
            "Defaults to current UTC time if neither --stop nor --window is set."
        ),
    )
    parser.add_argument(
        "--window",
        type=float,
        default=None,
        help="Time window in days from --start (alternative to --stop).",
    )
    parser.add_argument(
        "--kind",
        choices=["across", "within"],
        default="across",
        help=(
            "Fink date filter mode: 'across' (default) finds objects observed during the "
            "period regardless of first/last emission date; 'within' requires both first "
            "and last emission to be strictly inside the period."
        ),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help=(
            "Override conesearch radius in arcsec. Default: automatically computed from "
            "each pfsDesign's outermost fiber + --fov-margin (~2600 arcsec)."
        ),
    )
    parser.add_argument(
        "--fov-margin",
        type=float,
        default=DEFAULT_FOV_MARGIN_ARCSEC,
        help=(
            f"Outward margin in arcsec around the fiber convex hull to account for "
            f"cobra patrol radius (default: {DEFAULT_FOV_MARGIN_ARCSEC})."
        ),
    )
    parser.add_argument(
        "--cone-only",
        action="store_true",
        help="Keep all transients inside the search cone without filtering to the hexagonal FOV.",
    )
    parser.add_argument(
        "--exclude-sso",
        action="store_true",
        help="Exclude known Solar System Objects (f:is_sso == True).",
    )
    parser.add_argument(
        "--sky-reachable",
        action="store_true",
        help=(
            "Compute exact SKY-fiber cobra reachability (using pfs_utils/ics_cobraOps) "
            "and keep only transients physically reachable by a GOOD SKY cobra."
        ),
    )
    parser.add_argument(
        "--obstime",
        default=None,
        help="Observation time in UTC ISO format for sky->PFI transform (e.g. '2026-11-19T10:00:00Z').",
    )
    parser.add_argument(
        "--black-dot-margin",
        type=float,
        default=DEFAULT_BLACK_DOT_MARGIN,
        help=f"Black-dot avoidance radius margin (default: {DEFAULT_BLACK_DOT_MARGIN}).",
    )
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated Fink columns to request (overrides the default curated set).",
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Request all available Fink/LSST columns (slower).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path for matched transients.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional CSV path for per-pfsDesign summary table.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds (default: 120).",
    )
    return parser


def search_design(
    design: PfsDesignInfo,
    client: FinkLSSTClient,
    *,
    startdate: str,
    stopdate: str | None = None,
    window_days: float | None = None,
    kind: str = "across",
    radius_arcsec: float | None = None,
    fov_margin_arcsec: float = DEFAULT_FOV_MARGIN_ARCSEC,
    cone_only: bool = False,
    exclude_sso: bool = False,
    check_sky_reachability: bool = False,
    keep_only_sky_reachable: bool = False,
    bench=None,
    obs_time: str | None = None,
    black_dot_radius_margin: float = DEFAULT_BLACK_DOT_MARGIN,
    columns: list[str] | None = DEFAULT_CONESEARCH_COLUMNS,
) -> tuple[pd.DataFrame, dict]:
    """Query Fink/LSST for transients in a single pfsDesign FOV (and optionally SKY reachability)."""
    search_radius = (
        float(radius_arcsec)
        if radius_arcsec is not None
        else design.recommended_search_radius_arcsec(margin_arcsec=fov_margin_arcsec)
    )

    df = client.conesearch(
        ra=design.ra,
        dec=design.dec,
        radius_arcsec=search_radius,
        startdate=startdate,
        stopdate=stopdate,
        window_days=window_days,
        kind=kind,
        columns=columns,
    )

    n_cone = len(df)
    if n_cone == 0:
        summary = {
            "design_file": design.path.name,
            "design_name": design.design_name,
            "design_id": design.design_id_hex,
            "ra": design.ra,
            "dec": design.dec,
            "posang": design.posang,
            "n_sky_fibers": design.n_sky_fibers,
            "search_radius_arcsec": round(search_radius, 1),
            "n_in_cone": 0,
            "n_in_fov": 0,
            "n_sky_reachable": 0,
        }
        return df, summary

    ra_vals = pd.to_numeric(df["r:ra"], errors="coerce").to_numpy(dtype=float)
    dec_vals = pd.to_numeric(df["r:dec"], errors="coerce").to_numpy(dtype=float)

    df = df.copy()
    df["sep_deg"] = design.separation_deg(ra_vals, dec_vals)
    df["in_fov"] = design.contains_points(
        ra_vals, dec_vals, margin_arcsec=fov_margin_arcsec
    )

    if exclude_sso and "f:is_sso" in df.columns:
        df = df[df["f:is_sso"] != True].copy()  # noqa: E712

    n_in_cone_after = len(df)
    n_in_fov = int(df["in_fov"].sum())

    if not cone_only:
        df = df[df["in_fov"]].copy()

    df = df.reset_index(drop=True)

    n_sky_reachable = 0
    if (check_sky_reachability or keep_only_sky_reachable) and not df.empty:
        reach = check_sky_fiber_reachability(
            design.design,
            pd.to_numeric(df["r:ra"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["r:dec"], errors="coerce").to_numpy(dtype=float),
            bench=bench,
            obs_time=obs_time,
            black_dot_radius_margin=black_dot_radius_margin,
        )
        df = pd.concat([df, reach], axis=1)
        n_sky_reachable = int(df["sky_reachable"].sum())
        if keep_only_sky_reachable:
            df = df[df["sky_reachable"]].copy().reset_index(drop=True)

    # Attach pfsDesign metadata at the front of the table
    meta_cols = {
        "design_file": design.path.name,
        "design_name": design.design_name,
        "design_id": design.design_id_hex,
        "design_ra": design.ra,
        "design_dec": design.dec,
        "design_pa": design.posang,
    }
    for idx, (col_name, val) in enumerate(meta_cols.items()):
        df.insert(idx, col_name, val)

    if not df.empty:
        df = df.sort_values("sep_deg", ascending=True).reset_index(drop=True)

    summary = {
        "design_file": design.path.name,
        "design_name": design.design_name,
        "design_id": design.design_id_hex,
        "ra": design.ra,
        "dec": design.dec,
        "posang": design.posang,
        "n_sky_fibers": design.n_sky_fibers,
        "search_radius_arcsec": round(search_radius, 1),
        "n_in_cone": n_in_cone_after,
        "n_in_fov": n_in_fov,
        "n_sky_reachable": n_sky_reachable,
    }
    return df, summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.stopdate is not None and args.window is not None:
        parser.error("Specify either --stop or --window, not both.")

    try:
        design_paths = discover_pfs_designs(args.designs)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not design_paths:
        print("Error: No pfsDesign FITS files found.", file=sys.stderr)
        return 1

    if args.all_columns:
        columns: list[str] | None = None
    elif args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    else:
        columns = DEFAULT_CONESEARCH_COLUMNS

    bench = get_bench(black_dot_radius_margin=args.black_dot_margin) if args.sky_reachable else None
    client = FinkLSSTClient(timeout=args.timeout)
    all_results: list[pd.DataFrame] = []
    summaries: list[dict] = []

    print(
        f"Searching {len(design_paths)} pfsDesign file(s) "
        f"[start={args.startdate}, stop={args.stopdate or 'now'}, "
        f"window={args.window}, kind={args.kind}, sky_reachable={args.sky_reachable}]..."
    )

    for i, path in enumerate(design_paths, start=1):
        design = load_pfs_design(path)
        try:
            df, summary = search_design(
                design,
                client,
                startdate=args.startdate,
                stopdate=args.stopdate,
                window_days=args.window,
                kind=args.kind,
                radius_arcsec=args.radius,
                fov_margin_arcsec=args.fov_margin,
                cone_only=args.cone_only,
                exclude_sso=args.exclude_sso,
                check_sky_reachability=args.sky_reachable,
                keep_only_sky_reachable=args.sky_reachable,
                bench=bench,
                obs_time=args.obstime,
                black_dot_radius_margin=args.black_dot_margin,
                columns=columns,
            )
        except FinkAPIError as exc:
            print(
                f"[{i}/{len(design_paths)}] {design.design_name} ({path.name}): ERROR: {exc}",
                file=sys.stderr,
            )
            continue

        summaries.append(summary)
        if not df.empty:
            all_results.append(df)

        sky_str = f", sky_reachable={summary['n_sky_reachable']}" if args.sky_reachable else ""
        print(
            f"[{i}/{len(design_paths)}] {design.design_name} ({path.name}) "
            f"RA={design.ra:.4f} Dec={design.dec:+.4f} R={summary['search_radius_arcsec']:.0f}\" -> "
            f"cone={summary['n_in_cone']}, in_fov={summary['n_in_fov']}{sky_str}"
        )

    summary_df = pd.DataFrame(summaries)
    if args.summary_output:
        out_sum = Path(args.summary_output)
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_sum, index=False)
        print(f"Saved summary to {out_sum}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        n_unique = (
            combined["r:diaObjectId"].nunique()
            if "r:diaObjectId" in combined.columns
            else len(combined)
        )
        hit_col = "n_sky_reachable" if args.sky_reachable else "n_in_fov"
        n_designs_with_hits = int((summary_df[hit_col] > 0).sum()) if not summary_df.empty else 0
        print(
            f"\nTotal matched rows: {len(combined)} "
            f"({n_unique} unique diaObjectId across {n_designs_with_hits}/{len(summaries)} designs)"
        )
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(out_path, index=False)
            print(f"Saved transients to {out_path}")
    else:
        print("\nNo transients found in the specified pfsDesign FOV(s) and time range.")
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(out_path, index=False)
            print(f"Saved empty CSV to {out_path}")

    return 0


def build_select_sn_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="select_sn_targets",
        description=(
            "End-to-end Stage 1+2 pipeline: find Rubin/LSST transients in pfsDesign FOV(s), "
            "filter to exact SKY-fiber cobra reachability and >= min_nights observations, "
            "fetch cutouts on demand from Fink/LSST, and classify SN candidates using "
            "RubinTargetSelection (stamp_classifier)."
        ),
    )
    parser.add_argument(
        "designs",
        nargs="+",
        help="Path(s) to pfsDesign-*.fits file(s) or directory containing them.",
    )
    parser.add_argument(
        "--start",
        "--startdate",
        dest="startdate",
        required=True,
        help="Start date in UTC (e.g. '2026-11-01').",
    )
    parser.add_argument(
        "--stop",
        "--stopdate",
        dest="stopdate",
        default=None,
        help="Stop date in UTC (e.g. '2026-11-19'). Defaults to now.",
    )
    parser.add_argument(
        "--kind",
        choices=["across", "within"],
        default="across",
        help="Fink conesearch date filter mode (default: across).",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default="sn_output",
        help="Output directory for candidates.csv, design_sn_candidates.csv, candidates.pdf, etc.",
    )
    parser.add_argument(
        "--min-nights",
        type=int,
        default=2,
        help="Minimum number of distinct observing nights required (default: 2).",
    )
    parser.add_argument(
        "--min-nights-no-host",
        type=int,
        default=2,
        help="Minimum observing nights for no_host candidates (default: 2).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exclude step2 'uncertain' objects from SN candidates.",
    )
    parser.add_argument(
        "--include-negative-galaxy",
        action="store_true",
        help="Include (negative_real & galaxy) objects in SN candidates (excluded by default).",
    )
    parser.add_argument(
        "--keep-suspicious-visits",
        action="store_true",
        help="Do not reject sources from suspicious visits.",
    )
    parser.add_argument(
        "--per-night-band",
        choices=["snr", "latest"],
        default="snr",
        help="Which detection to classify per object/night/band (default: snr).",
    )
    parser.add_argument(
        "--bundle-dir",
        default=DEFAULT_BUNDLE_DIR,
        help=f"Path to frozen sn_pipeline bundle directory (default: {DEFAULT_BUNDLE_DIR}).",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CUTOUT_CACHE,
        help=f"Local cutout cache directory (default: {DEFAULT_CUTOUT_CACHE}).",
    )
    parser.add_argument(
        "--local-cutouts",
        default=DEFAULT_LOCAL_CUTOUTS,
        help=f"Optional existing local cutouts directory to reuse via symlink (default: {DEFAULT_LOCAL_CUTOUTS}).",
    )
    parser.add_argument(
        "--excluded-visits",
        default=DEFAULT_EXCLUDED_VISITS,
        help=f"Optional excluded_visits.csv path (default: {DEFAULT_EXCLUDED_VISITS}).",
    )
    parser.add_argument(
        "--obstime",
        default=None,
        help="Observation time in UTC ISO format for sky->PFI transform.",
    )
    parser.add_argument(
        "--black-dot-margin",
        type=float,
        default=DEFAULT_BLACK_DOT_MARGIN,
        help=f"Black-dot avoidance radius margin (default: {DEFAULT_BLACK_DOT_MARGIN}).",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Parallel threads for downloading Fink cutouts (default: 8).",
    )
    parser.add_argument(
        "--encode-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for SimCLR encoding (default: 4).",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip generating candidates.pdf.",
    )
    parser.add_argument(
        "--include-sso",
        action="store_true",
        help="Do not pre-filter known Solar System Objects (f:is_sso == True).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120).",
    )
    return parser


def select_sn_main(argv: list[str] | None = None) -> int:
    parser = build_select_sn_parser()
    args = parser.parse_args(argv)

    try:
        design_paths = discover_pfs_designs(args.designs)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not design_paths:
        print("Error: No pfsDesign FITS files found.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step A: Conesearch + Exact SKY-fiber Cobra Reachability ===")
    bench = get_bench(black_dot_radius_margin=args.black_dot_margin)
    client = FinkLSSTClient(timeout=args.timeout)

    reach_tables: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for i, path in enumerate(design_paths, start=1):
        design = load_pfs_design(path)
        try:
            df, summary = search_design(
                design,
                client,
                startdate=args.startdate,
                stopdate=args.stopdate,
                kind=args.kind,
                exclude_sso=not args.include_sso,
                check_sky_reachability=True,
                keep_only_sky_reachable=True,
                bench=bench,
                obs_time=args.obstime,
                black_dot_radius_margin=args.black_dot_margin,
            )
        except FinkAPIError as exc:
            print(
                f"[{i}/{len(design_paths)}] {design.design_name} ({path.name}): ERROR: {exc}",
                file=sys.stderr,
            )
            continue

        summaries.append(summary)
        if not df.empty:
            reach_tables.append(df)

        print(
            f"[{i}/{len(design_paths)}] {design.design_name} ({path.name}) "
            f"cone={summary['n_in_cone']}, in_fov={summary['n_in_fov']}, "
            f"sky_reachable={summary['n_sky_reachable']}"
        )

    pd.DataFrame(summaries).to_csv(out_dir / "design_summary.csv", index=False)

    if not reach_tables:
        print("\nNo SKY-fiber reachable transients found in the specified pfsDesign(s).")
        return 0

    reach_df = pd.concat(reach_tables, ignore_index=True)
    reach_df.to_csv(out_dir / "sky_reachable_transients.csv", index=False)

    # Cheap pre-filter: an object with r:nDiaSources < min_nights cannot have >= min_nights nights
    pre_df = reach_df
    if args.min_nights > 1 and "r:nDiaSources" in pre_df.columns:
        ndet = pd.to_numeric(pre_df["r:nDiaSources"], errors="coerce").fillna(0)
        pre_df = pre_df[ndet >= args.min_nights].copy()

    unique_oids = sorted(
        {int(x) for x in pd.to_numeric(pre_df["r:diaObjectId"], errors="coerce").dropna() if int(x) > 0}
    )
    print(
        f"\nSKY-reachable transients: {reach_df['r:diaObjectId'].nunique()} unique objects -> "
        f"{len(unique_oids)} objects with r:nDiaSources >= {args.min_nights}"
    )
    if not unique_oids:
        print("No objects pass the minimum detection count pre-filter.")
        return 0

    print("\n=== Step B: Fetch Full Alert Histories (/api/v1/sources) & Classify SN Candidates ===")
    all_sources = client.get_sources_for_classifier(unique_oids, batch_size=50)
    print(f"Fetched {len(all_sources)} total diaSources for {all_sources['diaObjectId'].nunique()} objects.")

    extra_roots = [args.local_cutouts] if args.local_cutouts else []
    candidates_df, _ = classify_sources(
        all_sources,
        start=args.startdate,
        end=args.stopdate,
        bundle_dir=args.bundle_dir,
        out_dir=out_dir,
        cache_root=args.cache_dir,
        extra_fits_roots=extra_roots,
        excluded_visits_path=args.excluded_visits,
        min_nights=args.min_nights,
        min_nights_no_host=args.min_nights_no_host,
        include_uncertain=not args.strict,
        include_negative_galaxy=args.include_negative_galaxy,
        keep_suspicious_visits=args.keep_suspicious_visits,
        per_night_band=args.per_night_band,
        download_workers=args.download_workers,
        encode_workers=args.encode_workers,
        make_pdf=not args.no_pdf,
    )

    if not candidates_df.empty:
        design_cols = [
            "design_file",
            "design_name",
            "design_id",
            "r:diaObjectId",
            "sep_deg",
            "pfi_x_mm",
            "pfi_y_mm",
            "n_sky_cobras",
            "sky_fiber_ids",
            "sky_cobra_ids",
            "nearest_assigned_mm",
        ]
        avail_cols = [c for c in design_cols if c in reach_df.columns]
        design_map = reach_df[avail_cols].rename(columns={"r:diaObjectId": "diaObjectId"})
        design_map["diaObjectId"] = design_map["diaObjectId"].astype("int64")
        merged = design_map.merge(candidates_df, on="diaObjectId", how="inner")
        merged = merged.sort_values(
            ["design_name", "s1_prob_positive_real"], ascending=[True, False]
        ).reset_index(drop=True)
        merged.to_csv(out_dir / "design_sn_candidates.csv", index=False)
        print(
            f"\n=== Done: {len(candidates_df)} unique SN candidates "
            f"({len(merged)} design-target pairs) saved to {out_dir}/ ==="
        )
    else:
        print(f"\n=== Done: 0 SN candidates passed classification (outputs in {out_dir}/) ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

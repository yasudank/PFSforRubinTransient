"""Command-line interface for finding Rubin/LSST transients in pfsDesign FOVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .fink_client import DEFAULT_CONESEARCH_COLUMNS, FinkAPIError, FinkLSSTClient
from .pfs_design import (
    DEFAULT_FOV_MARGIN_ARCSEC,
    PfsDesignInfo,
    discover_pfs_designs,
    load_pfs_design,
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
        help=(
            "Start date in UTC (e.g. '2026-11-01', '2026-11-01 00:00:00', or MJD/JD)."
        ),
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
    columns: list[str] | None = DEFAULT_CONESEARCH_COLUMNS,
) -> tuple[pd.DataFrame, dict]:
    """Query Fink/LSST for transients in a single pfsDesign FOV."""
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

    client = FinkLSSTClient(timeout=args.timeout)
    all_results: list[pd.DataFrame] = []
    summaries: list[dict] = []

    print(
        f"Searching {len(design_paths)} pfsDesign file(s) "
        f"[start={args.startdate}, stop={args.stopdate or 'now'}, "
        f"window={args.window}, kind={args.kind}]..."
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

        print(
            f"[{i}/{len(design_paths)}] {design.design_name} ({path.name}) "
            f"RA={design.ra:.4f} Dec={design.dec:+.4f} R={summary['search_radius_arcsec']:.0f}\" -> "
            f"cone={summary['n_in_cone']}, in_fov={summary['n_in_fov']}"
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
        n_designs_with_hits = int((summary_df["n_in_fov"] > 0).sum()) if not summary_df.empty else 0
        print(
            f"\nTotal matched rows: {len(combined)} "
            f"({n_unique} unique diaObjectId across {n_designs_with_hits}/{len(summaries)} designs with FOV hits)"
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


if __name__ == "__main__":
    raise SystemExit(main())


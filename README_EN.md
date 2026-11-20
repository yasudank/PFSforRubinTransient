# RubinTransient (`PFSforRubinTransient`)

[日本語 (Japanese)](README.md)

Tools for searching Rubin/LSST transients inside Subaru/PFS `pfsDesign` FITS fields of view via the [Fink/LSST object API](https://api.lsst.fink-portal.org/), checking exact `SKY`-fiber cobra reachability, classifying supernova candidates with [RubinTargetSelection](https://github.com/yasudank/RubinTargetSelection) (`stamp_classifier`), and reassigning `SKY` fibers with collision avoidance to produce new `pfsDesign` FITS files.

## Setup (`uv`)

The virtual environment is created according to `UV_PROJECT_ENVIRONMENT=.venv-$(uname -m)`.
Subaru PFS Project modules (`pfs-datamodel`, `pfs-utils`, `pfs-instdata`, `ics-cobraCharmer`, `ics-cobraOps`) are used for `pfsDesign` I/O and focal-plane / cobra collision geometry, and `yasudank/RubinTargetSelection` is installed from GitHub for supernova classification.

```bash
uv sync
```

### Updating Subaru/PFS Software Versions (`update-pfs-env`)

Resolve and update the Subaru/PFS package versions in `pyproject.toml` and the `uv` virtual environment from a YAML config file such as [config.yaml](config.yaml) (compatible with `yasudank/NetflowByUser` `config_*.yaml`).

```bash
# Preview resolved tags and whether pyproject.toml would change
uv run update-pfs-env config.yaml --dry-run

# Update pyproject.toml and run uv sync
uv run update-pfs-env config.yaml

# Pin exact floor tags (e.g., w.2026.39) instead of resolving latest tags for '>=' specs
uv run update-pfs-env config.yaml --exact-floor
```

## Stage 1: Search Transients in `pfsDesign` FOV (`find-transients`)

```bash
# Search one or more pfsDesign files or a directory (--sky-reachable restricts to SKY-cobra reachable objects)
uv run find-transients \
  /path/to/pfs_designs/pfsDesign-0x....fits \
  --start 2026-11-01 --stop 2026-11-19 \
  --exclude-sso --sky-reachable \
  -o transients.csv
```

## Stage 2: Select SKY-Reachable Supernova Candidates (`select-sn-targets`)

Runs on-demand against the Fink/LSST API without requiring a PostgreSQL database:

1. Queries Fink `/api/v1/conesearch` inside the hexagonal `pfsDesign` FOV (excluding known SSOs by default).
2. Evaluates exact reachability by `GOOD` `SKY` cobras (patrol ring `rMin <= r <= rMax` and black-dot avoidance) using `pfs_utils`, `ics_cobraOps`, and `pfs_instdata`.
3. Fetches full alert histories via Fink `/api/v1/sources` and filters to objects observed on at least 2 distinct nights (`--min-nights 2` by default).
4. Lazily downloads 30×30 multi-HDU FITS cutouts (`PRIMARY`, `UNCERT`, `MASKPLANE`, `PSFIMAGE`) from Fink `/api/v1/cutouts` in two stages (Difference for Step 1, then Science + Template for Step 1 survivors) and caches them locally (reusing `--local-cutouts` via symlinks if available).
5. Runs the frozen `RubinTargetSelection` (`stamp_classifier`) model bundle (`snp_v1`: Step 1 Difference SimCLR+RF → moving-object check → Step 2 Host morphology Group A + Group B RF) and writes `design_sn_candidates.csv`, `candidates.csv`, `candidates.pdf`, etc.

```bash
uv run select-sn-targets \
  /path/to/pfs_designs \
  --start 2026-11-01 --stop 2026-11-19 \
  --bundle-dir /path/to/models/sn_pipeline/snp_v1 \
  -o sn_output/run1
```

### Key Options (`select-sn-targets`)
- `--start` / `--stop`: Search date range (UTC)
- `--min-nights`: Minimum number of distinct observing nights (default: `2`)
- `--bundle-dir`: Path to the frozen `snp_v1` model bundle
- `--cache-dir`: Local cutout cache directory (default: `cache/cutouts`)
- `--local-cutouts`: Optional existing local cutout directory to reuse via symlinks (empty string disables)
- `--obstime`: Observation time in UTC ISO format for the sky-to-PFI transform
- `--strict`: Exclude Step 2 `uncertain` objects
- `--include-negative-galaxy`: Include `(negative_real & galaxy)` objects (excluded by default)
- `--no-pdf`: Skip generating `candidates.pdf`

## Stage 3: Reassign `SKY` Fibers and Write New `pfsDesign` FITS (`reassign-pfs-design`)

Reads `design_sn_candidates.csv` produced by `select-sn-targets`, verifies all safety checks below for each SN candidate in priority order, reassigns collision-free `SKY` fibers to `TargetType.SCIENCE`, and writes new `pfsDesign-0x....fits` files (`variant` incremented, `designId0` set to the original `pfsDesignId`, `pfsDesignId` recalculated via `calculate_pfsDesignId`, and `catId`/`proposalId` inherited from the original `SCIENCE` fibers).

- Bright-limit check: excludes transients whose latest `scienceFlux` AB magnitude (in any band) is `<= --min-mag` (default: `18.0` mag)
- Cobra patrol ring (`rMin <= r <= rMax`) and black-dot avoidance (`margin=1.65`)
- Fiducial avoidance (`>= 2.9 mm`)
- Minimum endpoint separation from other assigned cobras (`>= 2.1 mm`)
- Full `ics.cobraOps.CollisionSimulator` verification (**0 new endpoint collisions** and **0 new trajectory collisions**)

```bash
uv run reassign-pfs-design \
  --candidates sn_output/run1/design_sn_candidates.csv \
  --design-dir /path/to/pfs_designs \
  --min-mag 18.0 \
  -o modified_designs/run1
```

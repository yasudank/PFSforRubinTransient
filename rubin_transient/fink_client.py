"""Client for querying the Fink/LSST object API (https://api.lsst.fink-portal.org/)."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from tqdm import tqdm

FINK_LSST_API_URL = "https://api.lsst.fink-portal.org"
MAX_CONESEARCH_RADIUS_ARCSEC = 18000.0
NJY_TO_AB_ZP = 31.4  # AB mag = -2.5 * log10(flux_nJy) + 31.4

# Curated columns for transient discovery & downstream SN classification.
# Note: f:firstDiaSourceMjdTaiFink is required by Fink API when filtering by date.
DEFAULT_CONESEARCH_COLUMNS: list[str] = [
    "r:diaObjectId",
    "r:diaSourceId",
    "r:ra",
    "r:dec",
    "r:midpointMjdTai",
    "f:firstDiaSourceMjdTaiFink",
    "r:nDiaSources",
    "r:band",
    "r:psfFlux",
    "r:psfFluxErr",
    "r:scienceFlux",
    "r:scienceFluxErr",
    "r:snr",
    "r:reliability",
    "r:isDipole",
    "r:isNegative",
    "r:extendedness",
    "r:visit",
    "r:detector",
    "f:is_sso",
    "f:is_cataloged",
    "f:main_label_classifier",
    "f:main_label_crossmatch",
    "f:clf_snnSnVsOthers_score",
    "f:clf_earlySNIa_score",
    "f:xm_tns_fullname",
    "f:xm_tns_type",
    "f:xm_simbad_otype",
]

# Exact columns needed by stamp_classifier.sn_pipeline
SOURCES_PIPELINE_COLUMNS: list[str] = [
    "r:diaSourceId",
    "r:diaObjectId",
    "r:midpointMjdTai",
    "r:band",
    "r:ra",
    "r:dec",
    "r:visit",
    "r:scienceFlux",
    "r:scienceFluxErr",
    "r:psfFlux",
    "r:psfFluxErr",
]

_CUTOUT_KIND_MAP = {
    "difference": "Difference",
    "science": "Science",
    "template": "Template",
}

# Single-HDU 30x30 float32 FITS is 5760 bytes; multi-HDU LSST cutouts (with UNCERT/MASKPLANE/PSFIMAGE)
# are ~30-40 KB. Checking >= 10000 bytes detects legacy single-HDU cache files cheaply.
_MIN_MULTI_HDU_FITS_BYTES = 10000

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_date_str(value: str | None, *, end_of_day: bool = False) -> str | None:
    """Normalize a date string for the Fink/LSST API.

    If a date-only string ``YYYY-MM-DD`` is passed, appends ``00:00:00`` (or
    ``23:59:59`` when ``end_of_day=True``). Full timestamps, MJD, or JD strings
    are passed through unchanged.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if _DATE_ONLY_RE.match(s):
        return f"{s} 23:59:59" if end_of_day else f"{s} 00:00:00"
    return s


def flux_njy_to_ab_mag(flux_njy: pd.Series) -> pd.Series:
    """Convert flux in nJy to AB magnitude (NaN for non-positive or missing fluxes)."""
    arr = pd.to_numeric(flux_njy, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    pos = np.isfinite(arr) & (arr > 0.0)
    out[pos] = -2.5 * np.log10(arr[pos]) + NJY_TO_AB_ZP
    return pd.Series(out, index=flux_njy.index)


class FinkAPIError(RuntimeError):
    """Raised when the Fink/LSST API returns an error response."""


class FinkLSSTClient:
    """Thin client around the Fink/LSST REST API."""

    def __init__(
        self,
        base_url: str = FINK_LSST_API_URL,
        timeout: float = 120.0,
        max_retries: int = 5,
        retry_backoff: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.session = requests.Session()

    def _post_json_raw(self, endpoint: str, payload: dict):
        url = f"{self.base_url}{endpoint}"
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
                    continue
                raise FinkAPIError(f"Request to {url} failed: {exc}") from exc

            if resp.status_code == 200:
                if not resp.text.strip():
                    return []
                data = resp.json()
                if isinstance(data, dict) and data.get("status") == "error":
                    raise FinkAPIError(data.get("text", str(data)))
                return data

            if resp.status_code >= 500 and attempt < self.max_retries:
                time.sleep(self.retry_backoff * attempt)
                continue

            raise FinkAPIError(
                f"Fink API {endpoint} returned HTTP {resp.status_code}: {resp.text.strip()}"
            )

        raise FinkAPIError(f"Request to {url} failed after {self.max_retries} attempts: {last_exc}")

    def _post_json(self, endpoint: str, payload: dict) -> list[dict]:
        data = self._post_json_raw(endpoint, payload)
        if not data:
            return []
        if not isinstance(data, list):
            raise FinkAPIError(f"Unexpected response format from {endpoint}: {type(data)}")
        return data

    def conesearch(
        self,
        ra: float,
        dec: float,
        radius_arcsec: float,
        *,
        startdate: str | None = None,
        stopdate: str | None = None,
        window_days: float | None = None,
        kind: str = "across",
        columns: Sequence[str] | None = DEFAULT_CONESEARCH_COLUMNS,
        n: int | None = None,
    ) -> pd.DataFrame:
        """Query ``/api/v1/conesearch`` for LSST transients around (ra, dec)."""
        if radius_arcsec <= 0 or radius_arcsec > MAX_CONESEARCH_RADIUS_ARCSEC:
            raise ValueError(
                f"radius_arcsec must be in (0, {MAX_CONESEARCH_RADIUS_ARCSEC}], got {radius_arcsec}"
            )
        if kind not in ("across", "within"):
            raise ValueError(f"kind must be 'across' or 'within', got {kind!r}")
        if n is not None and (n <= 0 or n > 100):
            raise ValueError(f"Fink API requires n in [1, 100], got {n}")

        payload: dict[str, object] = {
            "ra": float(ra),
            "dec": float(dec),
            "radius": float(radius_arcsec),
            "kind": kind,
            "output-format": "json",
        }

        norm_start = normalize_date_str(startdate, end_of_day=False)
        norm_stop = normalize_date_str(stopdate, end_of_day=True)
        has_date_filter = norm_start is not None or norm_stop is not None or window_days is not None

        if norm_start is not None:
            payload["startdate"] = norm_start
        if window_days is not None:
            payload["window"] = float(window_days)
        elif norm_stop is not None:
            payload["stopdate"] = norm_stop
        if n is not None:
            payload["n"] = int(n)

        if columns is not None:
            col_list = list(columns)
            if has_date_filter and "f:firstDiaSourceMjdTaiFink" not in col_list:
                col_list.append("f:firstDiaSourceMjdTaiFink")
            for required_col in ("r:diaObjectId", "r:ra", "r:dec"):
                if required_col not in col_list:
                    col_list.insert(0, required_col)
            payload["columns"] = ",".join(col_list)

        records = self._post_json("/api/v1/conesearch", payload)
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "r:psfFlux" in df.columns:
            df["psf_mag"] = flux_njy_to_ab_mag(df["r:psfFlux"])
        if "r:scienceFlux" in df.columns:
            df["science_mag"] = flux_njy_to_ab_mag(df["r:scienceFlux"])
        return df

    def get_sources(
        self,
        dia_object_ids: Sequence[int | str],
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Query ``/api/v1/sources`` for full lightcurve alerts of given diaObjectIds."""
        if not dia_object_ids:
            return pd.DataFrame()
        payload: dict[str, object] = {
            "diaObjectId": ",".join(str(x) for x in dia_object_ids),
            "output-format": "json",
        }
        if columns is not None:
            payload["columns"] = ",".join(columns)
        records = self._post_json("/api/v1/sources", payload)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def get_sources_for_classifier(
        self,
        dia_object_ids: Sequence[int | str],
        *,
        batch_size: int = 50,
    ) -> pd.DataFrame:
        """Fetch full diaSource histories in batches, formatted for ``stamp_classifier``.

        Returns a DataFrame with columns:
        ``diaSourceId, diaObjectId, midpointMjdTai, band, ra, dec, visit,
        scienceFlux, scienceFluxErr, psfFlux, psfFluxErr``.
        """
        ids = sorted({int(x) for x in dia_object_ids if int(x) > 0})
        plain_cols = [c.split(":", 1)[1] for c in SOURCES_PIPELINE_COLUMNS]
        if not ids:
            return pd.DataFrame(columns=plain_cols)

        parts: list[pd.DataFrame] = []
        n_batches = (len(ids) + batch_size - 1) // batch_size
        iterator = range(0, len(ids), batch_size)
        if n_batches > 1:
            iterator = tqdm(iterator, total=n_batches, desc="Fetching /api/v1/sources")

        for i in iterator:
            chunk = ids[i : i + batch_size]
            df_chunk = self.get_sources(chunk, columns=SOURCES_PIPELINE_COLUMNS)
            if not df_chunk.empty:
                parts.append(df_chunk)

        if not parts:
            return pd.DataFrame(columns=plain_cols)

        df = pd.concat(parts, ignore_index=True)
        df = df.rename(columns={c: c.split(":", 1)[1] for c in df.columns if ":" in c})
        for col in plain_cols:
            if col not in df.columns:
                df[col] = np.nan

        df["diaSourceId"] = pd.to_numeric(df["diaSourceId"], errors="coerce").astype("Int64")
        df["diaObjectId"] = pd.to_numeric(df["diaObjectId"], errors="coerce").astype("Int64")
        df["visit"] = pd.to_numeric(df["visit"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["diaSourceId", "diaObjectId", "midpointMjdTai"]).copy()
        df["diaSourceId"] = df["diaSourceId"].astype(np.int64)
        df["diaObjectId"] = df["diaObjectId"].astype(np.int64)
        df["visit"] = df["visit"].fillna(0).astype(np.int64)
        for fcol in ("midpointMjdTai", "ra", "dec", "scienceFlux", "scienceFluxErr", "psfFlux", "psfFluxErr"):
            df[fcol] = pd.to_numeric(df[fcol], errors="coerce").astype(float)
        df["band"] = df["band"].astype(str)
        df = df.drop_duplicates("diaSourceId").reset_index(drop=True)
        return df[plain_cols]

    def fetch_cutout_fits(self, dia_source_id: int | str, image_type: str) -> bytes:
        """Fetch a full multi-HDU FITS cutout (PRIMARY, UNCERT, MASKPLANE, PSFIMAGE)."""
        kind = _CUTOUT_KIND_MAP[image_type]
        url = f"{self.base_url}/api/v1/cutouts"
        payload = {
            "diaSourceId": str(int(dia_source_id)),
            "kind": kind,
            "output-format": "FITS",
        }
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
                    continue
                raise FinkAPIError(f"Cutout request failed: {exc}") from exc

            if resp.status_code == 200 and resp.content.startswith(b"SIMPLE"):
                return resp.content

            if resp.status_code >= 500 and attempt < self.max_retries:
                time.sleep(self.retry_backoff * attempt)
                continue

            raise FinkAPIError(
                f"Fink API /api/v1/cutouts ({kind}) returned HTTP {resp.status_code}: "
                f"{resp.text[:200].strip()}"
            )

        raise FinkAPIError(f"Cutout request failed after {self.max_retries} attempts: {last_exc}")


def _is_valid_cached_fits(path: str, image_type: str) -> bool:
    if not os.path.exists(path):
        return False
    if image_type in ("science", "template"):
        # Require full multi-HDU FITS (with UNCERT and PSFIMAGE) for host_morphology
        try:
            return os.path.getsize(path) >= _MIN_MULTI_HDU_FITS_BYTES
        except OSError:
            return False
    return True


def ensure_cutouts_cached(
    src_df: pd.DataFrame,
    cache_root: str | Path,
    *,
    image_types: Sequence[str] = ("difference", "science", "template"),
    extra_fits_roots: Sequence[str | Path] = (),
    workers: int = 8,
    base_url: str = FINK_LSST_API_URL,
    timeout: float = 60.0,
) -> dict[str, int]:
    """Ensure full multi-HDU FITS cutouts exist in ``cache_root``.

    Uses the exact directory layout of ``stamp_classifier.dataset.get_cutout_path``.
    Reuses existing files in ``cache_root`` or symlinks from ``extra_fits_roots``
    before downloading missing cutouts in parallel from Fink ``/api/v1/cutouts``
    with ``output-format=FITS`` (preserving ``UNCERT``, ``MASKPLANE``, ``PSFIMAGE``).
    """
    import shutil
    from stamp_classifier.dataset import get_cutout_path

    cache_root_str = str(cache_root)
    extra_roots = [str(r) for r in extra_fits_roots if r and os.path.isdir(str(r))]
    req_types = tuple(image_types)

    stats = {"already_cached": 0, "linked_local": 0, "downloaded": 0, "failed": 0}
    to_download: list[tuple[int, float, str, int, tuple[str, ...]]] = []
    seen_sids: set[int] = set()

    for row in src_df.itertuples():
        sid = int(row.diaSourceId)
        if sid in seen_sids:
            continue
        seen_sids.add(sid)
        oid = int(row.diaObjectId)
        mjd = float(row.midpointMjdTai)
        band = str(row.band)

        dst_paths = {
            itype: get_cutout_path(cache_root_str, oid, mjd, band, sid, itype)
            for itype in req_types
        }
        missing = [itype for itype in req_types if not _is_valid_cached_fits(dst_paths[itype], itype)]
        if not missing:
            stats["already_cached"] += 1
            continue

        # Check local extra_fits_roots (e.g. /mnt/ugnas/work/LSST/fink2/cutouts)
        still_missing: list[str] = []
        linked_any = False
        for itype in missing:
            dp = dst_paths[itype]
            found_local = False
            for eroot in extra_roots:
                sp = get_cutout_path(eroot, oid, mjd, band, sid, itype)
                if _is_valid_cached_fits(sp, itype):
                    os.makedirs(os.path.dirname(dp), exist_ok=True)
                    if os.path.lexists(dp):
                        os.unlink(dp)
                    try:
                        os.symlink(os.path.abspath(sp), dp)
                    except OSError:
                        shutil.copy2(sp, dp)
                    linked_any = True
                    found_local = True
                    break
            if not found_local:
                still_missing.append(itype)

        if not still_missing:
            stats["linked_local"] += 1
        else:
            if linked_any:
                stats["linked_local"] += 1
            to_download.append((oid, mjd, band, sid, tuple(still_missing)))

    if not to_download:
        return stats

    def _download_one(item: tuple[int, float, str, int, tuple[str, ...]]) -> bool:
        oid, mjd, band, sid, missing_types = item
        client = FinkLSSTClient(base_url=base_url, timeout=timeout, max_retries=5, retry_backoff=3.0)
        ok = True
        for itype in missing_types:
            try:
                fits_bytes = client.fetch_cutout_fits(sid, itype)
            except Exception:
                ok = False
                continue
            dst = get_cutout_path(cache_root_str, oid, mjd, band, sid, itype)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = f"{dst}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(fits_bytes)
            os.replace(tmp, dst)
        return ok

    label = "+".join(req_types)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futures = [ex.submit(_download_one, item) for item in to_download]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Downloading Fink FITS ({label})",
        ):
            if fut.result():
                stats["downloaded"] += 1
            else:
                stats["failed"] += 1

    return stats

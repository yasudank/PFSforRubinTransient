"""Client for querying the Fink/LSST object API (https://api.lsst.fink-portal.org/)."""

from __future__ import annotations

import re
import time
from typing import Sequence

import numpy as np
import pandas as pd
import requests

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
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.session = requests.Session()

    def _post_json(self, endpoint: str, payload: dict) -> list[dict]:
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
                if not isinstance(data, list):
                    raise FinkAPIError(f"Unexpected response format from {url}: {type(data)}")
                return data

            if resp.status_code >= 500 and attempt < self.max_retries:
                time.sleep(self.retry_backoff * attempt)
                continue

            raise FinkAPIError(
                f"Fink API {endpoint} returned HTTP {resp.status_code}: {resp.text.strip()}"
            )

        raise FinkAPIError(f"Request to {url} failed after {self.max_retries} attempts: {last_exc}")

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
        """Query ``/api/v1/conesearch`` for LSST transients around (ra, dec).

        Parameters
        ----------
        ra, dec : float
            Cone center in decimal degrees (ICRS).
        radius_arcsec : float
            Search radius in arcseconds (max 18,000 arcsec = 5 deg).
        startdate : str, optional
            Start date in UTC (ISO ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS``, MJD, or JD).
        stopdate : str, optional
            Stop date in UTC. Ignored if ``window_days`` is provided.
        window_days : float, optional
            Time window in days from ``startdate`` (alternative to ``stopdate``).
        kind : {'across', 'within'}
            Time filter mode: ``'across'`` matches objects active during the date range;
            ``'within'`` requires both first and last emission to lie inside the range.
        columns : sequence of str or None
            Columns to request. Pass ``None`` to request all columns.
            When date filtering is used and ``columns`` is specified,
            ``f:firstDiaSourceMjdTaiFink`` is automatically included.
        n : int, optional
            Per-partition max object limit (API enforces <= 100 if specified).
        """
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
            # Always need coordinates and object ID for FOV matching
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


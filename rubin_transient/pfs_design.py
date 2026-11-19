"""Read pfsDesign FITS files via pfs.datamodel and evaluate PFS FOV geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from pfs.datamodel import FiberStatus, PfsDesign, TargetType

# Default PFS cobra patrol radius (~9.5 mm at ~10.9 arcsec/mm ≈ 104 arcsec)
DEFAULT_FOV_MARGIN_ARCSEC = 100.0
# Default PFS FOV circumradius fallback if no valid fiber coordinates are found
DEFAULT_PFS_RADIUS_DEG = 0.72


def _convex_hull_ccw(points: np.ndarray) -> np.ndarray:
    """Compute the 2D counter-clockwise convex hull using Andrew's monotone chain.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) with (x, y) coordinates.

    Returns
    -------
    np.ndarray
        Hull vertices in counter-clockwise order, shape (M, 2).
    """
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=float)


@dataclass
class PfsDesignInfo:
    """Summary, underlying PfsDesign object, and FOV geometry of a pfsDesign FITS file."""

    path: Path
    design: PfsDesign
    design_id: int
    design_id_hex: str
    design_name: str
    ra: float
    dec: float
    posang: float
    arms: str
    n_fibers_total: int
    n_fibers_valid: int
    n_sky_fibers: int
    max_fiber_sep_deg: float
    hull_xy_deg: np.ndarray

    @property
    def center(self) -> SkyCoord:
        return SkyCoord(self.ra * u.deg, self.dec * u.deg, frame="icrs")

    def recommended_search_radius_arcsec(
        self,
        margin_arcsec: float = DEFAULT_FOV_MARGIN_ARCSEC,
    ) -> float:
        """Bounding cone radius (arcsec) covering all fibers plus a patrol margin."""
        return float(self.max_fiber_sep_deg * 3600.0 + margin_arcsec)

    def separation_deg(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        """Angular separation (deg) of target coordinates from the design center."""
        ra_arr = np.asarray(ra_deg, dtype=float)
        dec_arr = np.asarray(dec_deg, dtype=float)
        if ra_arr.size == 0:
            return np.array([], dtype=float)
        coords = SkyCoord(ra_arr * u.deg, dec_arr * u.deg, frame="icrs")
        return np.asarray(self.center.separation(coords).deg, dtype=float)

    def contains_points(
        self,
        ra_deg: np.ndarray,
        dec_deg: np.ndarray,
        margin_arcsec: float = DEFAULT_FOV_MARGIN_ARCSEC,
    ) -> np.ndarray:
        """Check whether sky coordinates fall inside the hexagonal PFS FOV.

        Uses the 2D convex hull of valid fiber positions in tangent-plane offsets
        around (raBoresight, decBoresight), padded outward by ``margin_arcsec``.
        """
        ra_arr = np.asarray(ra_deg, dtype=float)
        dec_arr = np.asarray(dec_deg, dtype=float)
        if ra_arr.size == 0:
            return np.array([], dtype=bool)

        margin_deg = margin_arcsec / 3600.0
        coords = SkyCoord(ra_arr * u.deg, dec_arr * u.deg, frame="icrs")

        if len(self.hull_xy_deg) < 3:
            sep = np.asarray(self.center.separation(coords).deg, dtype=float)
            return sep <= (self.max_fiber_sep_deg + margin_deg)

        dra, ddec = self.center.spherical_offsets_to(coords)
        qx = np.asarray(dra.deg, dtype=float)
        qy = np.asarray(ddec.deg, dtype=float)

        hull = self.hull_xy_deg
        n_edges = len(hull)
        inside = np.ones(len(qx), dtype=bool)

        for i in range(n_edges):
            ax, ay = hull[i]
            bx, by = hull[(i + 1) % n_edges]
            dx = bx - ax
            dy = by - ay
            edge_len = float(np.hypot(dx, dy))
            if edge_len == 0.0:
                continue
            # For CCW hull, inward unit normal is (-dy / L, dx / L)
            signed_inward_dist = ((qx - ax) * (-dy) + (qy - ay) * dx) / edge_len
            inside &= signed_inward_dist >= -margin_deg

        return inside


def load_pfs_design(fits_path: str | Path) -> PfsDesignInfo:
    """Load a pfsDesign FITS file using ``pfs.datamodel.PfsDesign.readFits``."""
    path = Path(fits_path)
    design = PfsDesign.readFits(str(path))

    ra = float(design.raBoresight)
    dec = float(design.decBoresight)
    posang = float(design.posAng)
    design_name = str(design.designName).strip() if design.designName else path.stem
    design_id = int(design.pfsDesignId)
    design_id_hex = f"0x{design_id:016x}"
    arms = str(design.arms).strip() if design.arms else ""

    n_total = len(design)
    fiber_ra = np.asarray(design.ra, dtype=float)
    fiber_dec = np.asarray(design.dec, dtype=float)
    target_type = np.asarray(design.targetType, dtype=int)
    fiber_status = np.asarray(design.fiberStatus, dtype=int)

    valid = np.isfinite(fiber_ra) & np.isfinite(fiber_dec)
    n_valid = int(np.sum(valid))
    n_sky = int(
        np.sum(
            valid
            & (target_type == int(TargetType.SKY))
            & (fiber_status == int(FiberStatus.GOOD))
        )
    )

    center = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    if n_valid > 0:
        fiber_coords = SkyCoord(
            fiber_ra[valid] * u.deg,
            fiber_dec[valid] * u.deg,
            frame="icrs",
        )
        seps = np.asarray(center.separation(fiber_coords).deg, dtype=float)
        max_sep_deg = float(np.max(seps))
        dra, ddec = center.spherical_offsets_to(fiber_coords)
        xy = np.column_stack([dra.deg, ddec.deg])
        hull_xy = _convex_hull_ccw(xy)
    else:
        max_sep_deg = DEFAULT_PFS_RADIUS_DEG
        hull_xy = np.empty((0, 2), dtype=float)

    return PfsDesignInfo(
        path=path,
        design=design,
        design_id=design_id,
        design_id_hex=design_id_hex,
        design_name=design_name,
        ra=ra,
        dec=dec,
        posang=posang,
        arms=arms,
        n_fibers_total=n_total,
        n_fibers_valid=n_valid,
        n_sky_fibers=n_sky,
        max_fiber_sep_deg=max_sep_deg,
        hull_xy_deg=hull_xy,
    )


def discover_pfs_designs(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand file and directory arguments into a sorted list of pfsDesign FITS paths."""
    found: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            candidates = sorted(p.glob("pfsDesign-*.fits"))
            if not candidates:
                candidates = sorted(p.glob("*.fits"))
            for c in candidates:
                resolved = c.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(c)
        elif p.is_file():
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(p)
        else:
            raise FileNotFoundError(f"pfsDesign path not found: {p}")
    return found

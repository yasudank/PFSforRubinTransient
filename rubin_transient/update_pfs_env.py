"""Update Subaru-PFS package versions in pyproject.toml and uv venv from a PFS config YAML.

Supports config files like ``config_2026Nov.yaml`` from ``yasudank/NetflowByUser``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

PFS_PACKAGES: dict[str, dict[str, object]] = {
    "datamodel": {
        "dist_name": "pfs-datamodel",
        "repo_url": "https://github.com/Subaru-PFS/datamodel.git",
        "in_overrides": True,
    },
    "pfs_utils": {
        "dist_name": "pfs-utils",
        "repo_url": "https://github.com/Subaru-PFS/pfs_utils.git",
        "in_overrides": True,
    },
    "pfs_instdata": {
        "dist_name": "pfs-instdata",
        "repo_url": "https://github.com/Subaru-PFS/pfs_instdata.git",
        "in_overrides": True,
    },
    "ics_cobraCharmer": {
        "dist_name": "ics-cobracharmer",
        "repo_url": "https://github.com/Subaru-PFS/ics_cobraCharmer.git",
        "in_overrides": True,
    },
    "ics_cobraOps": {
        "dist_name": "ics-cobraops",
        "repo_url": "https://github.com/Subaru-PFS/ics_cobraOps.git",
        "in_overrides": False,
    },
}

_CONSTRAINT_RE = re.compile(r"^\s*(>=|<=|>|<|==)?\s*(.+?)\s*$")
_WEEKLY_RE = re.compile(r"^w\.(\d{4})\.(\d+)(?:\.(\d+))?$")
_SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)(?:\.(\d+))?$")


def parse_constraint(spec: str) -> tuple[str, str]:
    """Split a constraint like '>=w.2026.39' into (operator, version_str)."""
    m = _CONSTRAINT_RE.match(str(spec))
    if not m:
        return "==", str(spec).strip()
    op = m.group(1) or "=="
    ver = m.group(2).strip()
    return op, ver


def _parse_version_tuple(tag: str) -> tuple[str, tuple[int, ...]] | None:
    """Parse a weekly tag ('w.2026.39') or semver tag ('v2.0.9', '1.8.97')."""
    mw = _WEEKLY_RE.match(tag)
    if mw:
        nums = tuple(int(x) for x in mw.groups() if x is not None)
        return ("weekly", nums)
    ms = _SEMVER_RE.match(tag)
    if ms:
        nums = tuple(int(x) for x in ms.groups() if x is not None)
        return ("semver", nums)
    return None


def _compare_versions(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    maxlen = max(len(a), len(b))
    pa = a + (0,) * (maxlen - len(a))
    pb = b + (0,) * (maxlen - len(b))
    return (pa > pb) - (pa < pb)


def _matches_constraint(tag: str, op: str, target_ver: str) -> bool:
    if op == "==":
        return tag == target_ver
    pt = _parse_version_tuple(tag)
    pv = _parse_version_tuple(target_ver)
    if pt is None or pv is None or pt[0] != pv[0]:
        return False
    cmp = _compare_versions(pt[1], pv[1])
    if op == ">=":
        return cmp >= 0
    if op == ">":
        return cmp > 0
    if op == "<=":
        return cmp <= 0
    if op == "<":
        return cmp < 0
    return False


def list_remote_tags(repo_url: str) -> list[str]:
    """Return tag names from a remote Git repository via ``git ls-remote --tags``."""
    proc = subprocess.run(
        ["git", "ls-remote", "--tags", "--refs", repo_url],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    tags: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].startswith("refs/tags/"):
            tags.append(parts[1][len("refs/tags/") :])
    return tags


def resolve_git_ref(
    package_key: str,
    spec: str,
    repo_url: str,
    *,
    exact_floor: bool = False,
) -> str:
    """Resolve a version spec from config YAML to a concrete Git tag/branch/rev."""
    op, target_ver = parse_constraint(spec)
    if op == "==" or exact_floor:
        return target_ver

    tags = list_remote_tags(repo_url)
    matching = [t for t in tags if _matches_constraint(t, op, target_ver)]
    if not matching:
        print(
            f"Warning: No remote tags matched '{spec}' for {package_key}; "
            f"falling back to '{target_ver}'.",
            file=sys.stderr,
        )
        return target_ver

    def _sort_key(t: str) -> tuple[int, ...]:
        parsed = _parse_version_tuple(t)
        assert parsed is not None
        return parsed[1] + (0,) * (4 - len(parsed[1]))

    matching.sort(key=_sort_key, reverse=True)
    return matching[0]


def update_pyproject_toml(
    pyproject_path: Path,
    resolved_refs: dict[str, str],
    *,
    dry_run: bool = False,
) -> bool:
    """Update ``[tool.uv.sources]`` and ``override-dependencies`` in pyproject.toml.

    Returns True if the file content changed.
    """
    text = pyproject_path.read_text(encoding="utf-8")
    new_text = text

    for pkg_key, ref in resolved_refs.items():
        meta = PFS_PACKAGES[pkg_key]
        dist_name = str(meta["dist_name"])
        repo_url = str(meta["repo_url"])

        # Update [tool.uv.sources] line:
        #   pfs-datamodel = { git = "https://github.com/Subaru-PFS/datamodel.git", tag = "..." }
        src_pattern = re.compile(
            rf'^({re.escape(dist_name)}\s*=\s*\{{\s*git\s*=\s*"{re.escape(repo_url)}"\s*,\s*)(?:tag|branch|rev)\s*=\s*"[^"]*"(\s*\}})$',
            re.MULTILINE,
        )
        new_text, n_src = src_pattern.subn(rf'\1tag = "{ref}"\2', new_text)
        if n_src == 0:
            print(
                f"Warning: Could not find [tool.uv.sources] entry for {dist_name} in {pyproject_path}",
                file=sys.stderr,
            )

        # Update override-dependencies line if present:
        #   "pfs-datamodel @ git+https://github.com/Subaru-PFS/datamodel.git@...",
        if meta["in_overrides"]:
            ov_pattern = re.compile(
                rf'("{re.escape(dist_name)}\s*@\s*git\+{re.escape(repo_url)}@)[^"]+(")',
            )
            new_text, _ = ov_pattern.subn(rf"\g<1>{ref}\g<2>", new_text)

    changed = new_text != text
    if changed and not dry_run:
        pyproject_path.write_text(new_text, encoding="utf-8")
    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="update-pfs-env",
        description=(
            "Update Subaru-PFS package versions in pyproject.toml and sync the uv "
            "virtual environment from a PFS config YAML (e.g. config_2026Nov.yaml)."
        ),
    )
    parser.add_argument(
        "config_yaml",
        help="Path to PFS config YAML file (e.g. /mnt/ugnas/work/PFS/galuda/NetflowByUser/config_2026Nov.yaml).",
    )
    parser.add_argument(
        "--pyproject",
        default=None,
        help="Path to pyproject.toml (default: ./pyproject.toml).",
    )
    parser.add_argument(
        "--exact-floor",
        action="store_true",
        help="For '>=' constraints, pin the exact floor tag (e.g. w.2026.39) instead of resolving the latest remote tag.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve versions and show changes without modifying pyproject.toml or syncing uv.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Update pyproject.toml without running 'uv sync'.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config_yaml)
    if not config_path.is_file():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    pyproject_path = (
        Path(args.pyproject)
        if args.pyproject
        else Path(__file__).resolve().parent.parent / "pyproject.toml"
    )
    if not pyproject_path.is_file():
        pyproject_path = Path.cwd() / "pyproject.toml"
    if not pyproject_path.is_file():
        print(f"Error: pyproject.toml not found: {pyproject_path}", file=sys.stderr)
        return 1

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pfs_cfg = cfg.get("pfs") or {}
    if not isinstance(pfs_cfg, dict):
        print(f"Error: Missing or invalid 'pfs:' section in {config_path}", file=sys.stderr)
        return 1

    print(f"Reading PFS versions from {config_path} (run={cfg.get('run', 'unknown')})...")
    resolved: dict[str, str] = {}
    for pkg_key, meta in PFS_PACKAGES.items():
        spec = pfs_cfg.get(pkg_key)
        if not spec:
            print(f"  {pkg_key:<18} -> (not specified in config, keeping current)")
            continue
        ref = resolve_git_ref(
            pkg_key,
            str(spec),
            str(meta["repo_url"]),
            exact_floor=args.exact_floor,
        )
        resolved[pkg_key] = ref
        print(f"  {pkg_key:<18} spec={str(spec):<14} -> {ref}")

    if not resolved:
        print("No PFS package versions found to update.")
        return 0

    changed = update_pyproject_toml(pyproject_path, resolved, dry_run=args.dry_run)
    if args.dry_run:
        print(f"\n[dry-run] pyproject.toml would be {'updated' if changed else 'unchanged'}.")
        return 0

    if changed:
        print(f"\nUpdated {pyproject_path}")
    else:
        print(f"\n{pyproject_path} already matches the resolved versions.")

    if args.no_sync:
        return 0

    project_dir = pyproject_path.parent
    cmd = ["uv", "sync"]
    for pkg_key in resolved:
        dist_name = str(PFS_PACKAGES[pkg_key]["dist_name"])
        cmd.extend(["--upgrade-package", dist_name])

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(project_dir))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

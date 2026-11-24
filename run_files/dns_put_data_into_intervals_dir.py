#!/usr/bin/env python3
"""
Collect mean-velocity and Reynolds-stress profiles from DNS interval runs.

Expected input layout:
    runs/dns/
        boxcar_dns_dt_0.001/
            t0_308.01_tf_309.01/
                mean_profiles/mean_w_309.01.npy
                reynolds_profiles/reynolds_stress_profile_309.01.npy
            t0_309.01_tf_310.01/
                ...

Output:
    runs/dns/dns_intervals_dt_0.001/
        w_mean_308.01_309.01.npy
        ww_308.01_309.01.npy
        ...

The interval start/end values are taken from each interval directory name.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


INTERVAL_PATTERN = re.compile(
    r"^t0_(?P<t_start>\d+(?:\.\d+)?)_tf_(?P<t_end>\d+(?:\.\d+)?)$"
)


def collect_dns_interval_profiles(
    dns_runs_dir: Path,
    overwrite: bool = False,
) -> None:
    """Collect profiles from every interval run for the requested DNS timestep."""
    output_dir = dns_runs_dir / "dns_intervals"
    output_dir.mkdir(parents=True, exist_ok=True)

    # The normal run directory is e.g. boxcar_dns_dt_0.001.
    run_dirs = sorted(
        path
        for path in dns_runs_dir.iterdir()
        if path.is_dir() and path.name != output_dir.name
    )

    if not run_dirs:
        raise FileNotFoundError(f"No DNS run directorieswere found in {dns_runs_dir}")

    copied = 0
    skipped = 0
    missing = 0

    for interval_dir in run_dirs:
        match = INTERVAL_PATTERN.match(interval_dir.name)
        if match is None:
            continue

        t_start = match.group("t_start")
        t_end = match.group("t_end")

        mean_source = interval_dir / "mean_profiles" / f"mean_w_{t_end}.npy"
        reynolds_source = (
            interval_dir / "reynolds_profiles" / f"reynolds_stress_profile_{t_end}.npy"
        )

        # Also tolerate equivalent float formatting in the profile filenames.
        if not mean_source.exists():
            candidates = sorted((interval_dir / "mean_profiles").glob("mean_w_*.npy"))
            mean_source = candidates[0] if len(candidates) == 1 else mean_source

        if not reynolds_source.exists():
            candidates = sorted(
                (interval_dir / "reynolds_profiles").glob(
                    "reynolds_stress_profile_*.npy"
                )
            )
            reynolds_source = candidates[0] if len(candidates) == 1 else reynolds_source

        if not mean_source.exists() or not reynolds_source.exists():
            print(
                f"[Missing] {interval_dir.name}: "
                f"mean={mean_source.exists()}, "
                f"Reynolds stress={reynolds_source.exists()}"
            )
            missing += 1
            continue

        mean_destination = output_dir / f"w_mean_{t_start}_{t_end}.npy"
        reynolds_destination = output_dir / f"ww_{t_start}_{t_end}.npy"

        destinations = (
            (mean_source, mean_destination),
            (reynolds_source, reynolds_destination),
        )

        if not overwrite and any(dst.exists() for _, dst in destinations):
            print(
                f"[Skipped] {interval_dir.name}: output already exists "
                "(use --overwrite to replace it)"
            )
            skipped += 1
            continue

        mean_profile = np.load(mean_source)
        reynolds_profile = np.load(reynolds_source)

        np.save(mean_destination, mean_profile)
        np.save(reynolds_destination, reynolds_profile)

        print(f"[Saved] {mean_destination.name} and {reynolds_destination.name}")
        copied += 1

    print(
        f"\nFinished: {copied} interval(s) copied, "
        f"{skipped} skipped, {missing} with missing profiles."
    )
    print(f"Output directory: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect DNS profiles into a flat interval-reference directory."
    )
    parser.add_argument(
        "--dns-runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "DNS" / "filtered",
        help="Parent directory containing the DNS run folders "
        "(default: project_root/runs/dns, relative to this script).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files that already exist.",
    )
    args = parser.parse_args()

    collect_dns_interval_profiles(
        dns_runs_dir=args.dns_runs_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

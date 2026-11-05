#!/usr/bin/env python3
"""Create Ottawa population, household, and family layers for census years."""

from __future__ import annotations

import argparse
from pathlib import Path

from create_ottawa_household_layers import generate_year as generate_household_layers
from create_ottawa_population_layer import (
    CENSUS_SPECS,
    generate_population_layer,
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=project_dir,
        help="Project root containing data/ and outputs/.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(CENSUS_SPECS),
        help="Census years to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing population and household outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unsupported = [year for year in args.years if year not in CENSUS_SPECS]
    if unsupported:
        raise SystemExit(f"Unsupported census years: {unsupported}")

    project_dir = args.project_dir.resolve()
    land_use_metadata_path = (
        project_dir / "data" / "LU Ottawa rst" / "LU2020 Ottawa_b1.rdc"
    )

    for year in dict.fromkeys(args.years):
        spec = CENSUS_SPECS[year]
        print(f"\n=== Processing Ottawa layers for {year} ===")
        print("Population layers")
        generate_population_layer(
            spec,
            project_dir,
            land_use_metadata_path,
            args.overwrite,
        )
        print("Household and family layers")
        generate_household_layers(
            year,
            project_dir,
            land_use_metadata_path,
            args.overwrite,
        )


if __name__ == "__main__":
    main()

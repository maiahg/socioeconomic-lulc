#!/usr/bin/env python3
"""Create standardized Ottawa household and family layers for census years."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from create_ottawa_population_layer import (
    CENSUS_SPECS,
    IDRISI_REFERENCE_SYSTEM,
    NODATA,
    TARGET_CRS,
    rasterize_polygons,
    read_idrisi_metadata,
    remove_if_allowed,
    write_idrisi_real,
    write_optional_geotiff,
)


IVT_CODES = {
    2001: {
        "population_total": [1],
        "children_0_14": [4, 5, 6, 23, 24, 25],
        "average_household_size": [125],
        "households_total": [118],
        "households_1_person": [119],
        "households_2_person": [120],
        "households_3_person": [121],
        "non_family_households": [129],
    },
    2006: {
        "population_total": [1],
        "children_0_14": [4, 5, 6, 23, 24, 25],
        "average_household_size": [135],
        "households_total": [128],
        "households_1_person": [129],
        "households_2_person": [130],
        "households_3_person": [131],
        "non_family_households": [139],
    },
}

CSV_CODES = {
    2016: {
        "population_total": [1],
        "children_0_14": [9],
        "average_household_size": [58],
        "households_total": [51],
        "households_1_person": [52],
        "households_2_person": [53],
        "households_3_person": [54],
        "non_family_households": [97],
    },
    2021: {
        "population_total": [1],
        "children_0_14": [9],
        "average_household_size": [57],
        "households_total": [50],
        "households_1_person": [51],
        "households_2_person": [52],
        "households_3_person": [53],
        "non_family_households": [109, 110],
    },
}


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
        help="Replace existing household outputs.",
    )
    return parser.parse_args()


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().replace(",", "")
    if not value or value in {"...", "x", "X", "-"}:
        return None
    try:
        return float(value)
    except ValueError:
        # Statistics Canada uses flags such as F (not publishable), E
        # (estimated), and similar symbols in numeric CSV columns.
        return None


def add_metric(record: dict[str, float | None], metric: str, value: float | None) -> None:
    """Add a metric while preserving suppression/missing-value information."""
    if metric not in record:
        record[metric] = value
    elif record[metric] is None or value is None:
        record[metric] = None
    else:
        record[metric] += value


def extract_ivt_metrics(year: int, path: Path) -> pd.DataFrame:
    try:
        import pyivt
    except ImportError as exc:
        raise RuntimeError(
            "IVT processing requires pyivt and pyarrow. Install them with "
            "'.venv/bin/python -m pip install pyivt pyarrow'."
        ) from exc

    ivt = pyivt.open(path)
    member_tables = ivt.to_member_tables()
    profile_dimension = next(
        key
        for key, table in member_tables.items()
        if "label_en" in table.column_names
        and table.to_pandas()["label_en"]
        .astype(str)
        .str.match(fr"^Population,\s*{year}\b")
        .any()
    )
    geography_dimension = "geography"
    geography = member_tables[geography_dimension].to_pandas()
    geography["DAUID"] = geography["label_en"].astype(str).str.extract(
        r"(3506\d{4})", expand=False
    )
    geography = geography[geography["DAUID"].notna()].copy()

    codes = IVT_CODES[year]
    selected_codes = sorted({code for values in codes.values() for code in values})
    selection = {
        profile_dimension: selected_codes,
        geography_dimension: geography["code"].astype(int).tolist(),
    }
    values = ivt.to_arrow(selection=selection).to_pydict()
    result = pd.DataFrame(
        {
            "profile_code": values[profile_dimension],
            "geography_code": values[geography_dimension],
            "value": values["value"],
        }
    )
    value_lookup = {
        (int(row.profile_code), int(row.geography_code)): row.value
        for row in result.itertuples(index=False)
    }

    def sum_ivt_values(profile_codes: list[int], geography_code: int) -> float:
        selected = [
            value_lookup.get((code, geography_code)) for code in profile_codes
        ]
        if any(value is None or pd.isna(value) for value in selected):
            return np.nan
        return float(sum(selected))

    records: list[dict[str, object]] = []
    for geography_row in geography.itertuples(index=False):
        record: dict[str, object] = {"DAUID": geography_row.DAUID}
        for metric, metric_codes in codes.items():
            record[metric] = sum_ivt_values(metric_codes, int(geography_row.code))
        records.append(record)
    return pd.DataFrame(records)


def extract_2011_metrics(path: Path) -> pd.DataFrame:
    records: dict[str, dict[str, float | None]] = {}
    child_labels = {"0 to 4 years", "5 to 9 years", "10 to 14 years"}
    household_topic = "Household and dwelling characteristics"

    with path.open("r", encoding="cp1252", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            dauid = row["Geo_Code"].strip()
            if len(dauid) != 8 or not dauid.startswith("3506"):
                continue
            record = records.setdefault(dauid, {})
            topic = row["Topic"].strip()
            characteristic = row["Characteristic"].strip()
            value = parse_number(row["Total"])

            if characteristic == "Population in 2011":
                record["population_total"] = value
            elif topic == "Age characteristics" and characteristic in child_labels:
                add_metric(record, "children_0_14", value)
            elif (
                topic == household_topic
                and characteristic == "Average number of persons in private households"
            ):
                record["average_household_size"] = value
            elif (
                topic == household_topic
                and characteristic
                == "Total number of private households by household size"
            ):
                record["households_total"] = value
            elif topic == household_topic and characteristic == "1 person":
                record["households_1_person"] = value
            elif topic == household_topic and characteristic == "2 persons":
                record["households_2_person"] = value
            elif topic == household_topic and characteristic == "3 persons":
                record["households_3_person"] = value
            elif topic == household_topic and characteristic == "Non-census-family households":
                record["non_family_households"] = value

    return pd.DataFrame.from_dict(records, orient="index").rename_axis("DAUID").reset_index()


def extract_csv_metrics(year: int, path: Path) -> pd.DataFrame:
    if year == 2011:
        return extract_2011_metrics(path)

    spec = CENSUS_SPECS[year]
    codes = CSV_CODES[year]
    records: dict[str, dict[str, float | None]] = {}
    with path.open("r", encoding=spec.encoding, newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            dauid = row[spec.dauid_column].strip()
            if len(dauid) != 8 or not dauid.startswith("3506"):
                continue
            id_column = (
                "Member ID: Profile of Dissemination Areas (2247)"
                if year == 2016
                else "CHARACTERISTIC_ID"
            )
            characteristic_id = int(row[id_column])
            metric_names = [
                metric for metric, ids in codes.items() if characteristic_id in ids
            ]
            if not metric_names:
                continue
            record = records.setdefault(dauid, {})
            value = parse_number(row[spec.population_column])
            for metric in metric_names:
                add_metric(record, metric, value)

    return pd.DataFrame.from_dict(records, orient="index").rename_axis("DAUID").reset_index()


def standardize_metrics(metrics: pd.DataFrame, year: int) -> pd.DataFrame:
    required = [
        "population_total",
        "children_0_14",
        "average_household_size",
        "households_total",
        "households_1_person",
        "households_2_person",
        "households_3_person",
        "non_family_households",
    ]
    missing = [column for column in required if column not in metrics.columns]
    if missing:
        raise ValueError(f"{year} household data is missing fields: {missing}")

    for column in required:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    missing_rows = metrics.loc[metrics[required].isna().any(axis=1), "DAUID"].tolist()
    if missing_rows:
        print(
            f"Warning: {year} household data has missing/suppressed values for "
            f"{len(missing_rows)} DAs; those raster cells will be NODATA."
        )

    metrics["households_4_plus"] = (
        metrics["households_total"]
        - metrics["households_1_person"]
        - metrics["households_2_person"]
        - metrics["households_3_person"]
    )
    metrics["family_households"] = (
        metrics["households_total"] - metrics["non_family_households"]
    )
    metrics["children_0_14_share"] = (
        metrics["children_0_14"] / metrics["population_total"].replace(0, np.nan)
    )
    for category in ("1_person", "2_person", "3_person", "4_plus"):
        metrics[f"households_{category}_share"] = (
            metrics[f"households_{category}"]
            / metrics["households_total"].replace(0, np.nan)
        )
    metrics["family_households_share"] = (
        metrics["family_households"]
        / metrics["households_total"].replace(0, np.nan)
    )
    metrics["non_family_households_share"] = (
        metrics["non_family_households"]
        / metrics["households_total"].replace(0, np.nan)
    )
    return metrics


def generate_year(
    year: int,
    project_dir: Path,
    land_use_metadata_path: Path,
    overwrite: bool,
) -> None:
    spec = CENSUS_SPECS[year]
    raw_path = project_dir / spec.raw_census_path
    boundary_path = project_dir / spec.boundary_path
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not boundary_path.exists():
        raise FileNotFoundError(boundary_path)
    if not land_use_metadata_path.exists():
        raise FileNotFoundError(land_use_metadata_path)

    if spec.source_format == "ivt":
        metrics = extract_ivt_metrics(year, raw_path)
    else:
        metrics = extract_csv_metrics(year, raw_path)
    metrics = standardize_metrics(metrics, year)

    read_kwargs = {}
    if spec.boundary_layer is not None:
        read_kwargs["layer"] = spec.boundary_layer
    if spec.boundary_bbox is not None:
        read_kwargs["bbox"] = spec.boundary_bbox
    da = gpd.read_file(boundary_path, **read_kwargs)
    da["DAUID"] = da["DAUID"].astype(str).str.strip()
    da = da[da["DAUID"].str.startswith("3506")].copy()
    da = da.merge(metrics, on="DAUID", how="left", validate="one_to_one")

    metric_columns = [
        "population_total",
        "children_0_14",
        "children_0_14_share",
        "average_household_size",
        "households_total",
        "households_1_person_share",
        "households_2_person_share",
        "households_3_person_share",
        "households_4_plus_share",
        "family_households_share",
        "non_family_households_share",
    ]
    missing = da[metric_columns].isna().any(axis=1)
    if missing.any():
        print(
            f"Warning: {year} household metrics are unavailable for "
            f"{missing.sum()} boundary DAs; those raster cells will be NODATA."
        )

    da = da.to_crs(TARGET_CRS)
    da["area_km2"] = da.geometry.area / 1_000_000
    output_dir = project_dir / "outputs" / "ottawa_households" / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = output_dir / f"ottawa_household_metrics_{year}.gpkg"
    summary_path = output_dir / f"ottawa_household_metrics_{year}_summary.json"

    raster_specs = {
        "average_household_size": "direct",
        "households_total": "count",
        "households_1_person_share": "direct",
        "households_2_person_share": "direct",
        "households_3_person_share": "direct",
        "households_4_plus_share": "direct",
        "children_0_14": "count",
        "children_0_14_share": "direct",
        "family_households_share": "direct",
        "non_family_households_share": "direct",
    }
    output_paths = [gpkg_path, summary_path]
    for metric_name in raster_specs:
        output_paths.extend(
            [
                output_dir / f"ottawa_{metric_name}_{year}.rst",
                output_dir / f"ottawa_{metric_name}_{year}.rdc",
                output_dir / f"ottawa_{metric_name}_{year}.tif",
            ]
        )
    for path in output_paths:
        remove_if_allowed(path, overwrite)

    rdc = read_idrisi_metadata(land_use_metadata_path)
    width = int(rdc["columns"])
    height = int(rdc["rows"])
    resolution = float(rdc["resolution"])
    min_x = float(rdc["min. x"])
    max_x = float(rdc["max. x"])
    min_y = float(rdc["min. y"])
    max_y = float(rdc["max. y"])
    cell_area_km2 = resolution * resolution / 1_000_000
    geometries = np.asarray(da.geometry.to_numpy(), dtype=object)
    validation: dict[str, dict[str, float]] = {}

    for metric_name, metric_kind in raster_specs.items():
        values = da[metric_name].to_numpy(dtype="float32")
        if metric_kind == "count":
            values = values / da["area_km2"].to_numpy(dtype="float32")
        values[~np.isfinite(values)] = NODATA
        array = rasterize_polygons(
            geometries,
            values,
            width=width,
            height=height,
            resolution=resolution,
            min_x=min_x,
            max_y=max_y,
            fill=NODATA,
        )
        if metric_kind == "count":
            array = np.where(array == NODATA, NODATA, array * cell_area_km2).astype(
                "float32"
            )
            valid = array[array != NODATA]
            expected = float(da[metric_name].sum())
            actual = float(valid.sum())
            validation[metric_name] = {
                "expected_da_total": expected,
                "raster_total": actual,
                "difference_pct": 100 * (actual - expected) / expected
                if expected
                else 0.0,
            }

        valid = array[array != NODATA]
        rst_path = output_dir / f"ottawa_{metric_name}_{year}.rst"
        rdc_path = output_dir / f"ottawa_{metric_name}_{year}.rdc"
        write_idrisi_real(
            array,
            rst_path,
            rdc_path,
            width=width,
            height=height,
            resolution=resolution,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            title=f"{year} Ottawa {metric_name}",
            display_min=float(valid.min()),
            display_max=float(valid.max()),
        )
        write_optional_geotiff(
            output_dir / f"ottawa_{metric_name}_{year}.tif",
            array,
            width=width,
            height=height,
            resolution=resolution,
            min_x=min_x,
            max_y=max_y,
        )

    da.to_file(
        gpkg_path,
        layer=f"ottawa_household_metrics_{year}",
        driver="GPKG",
    )
    summary = {
        "census_year": year,
        "source_census_file": str(raw_path),
        "source_boundary_file": str(boundary_path),
        "land_use_grid_metadata": str(land_use_metadata_path),
        "target_crs": TARGET_CRS,
        "idrisi_reference_system": IDRISI_REFERENCE_SYSTEM,
        "ottawa_da_count": int(len(da)),
        "raster_metrics": list(raster_specs),
        "count_raster_validation": validation,
        "children_definition": "population aged 0 to 14",
        "household_size_definition": "one, two, three, and four-or-more persons",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Created household layers for {year}: {output_dir}")


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
        generate_year(year, project_dir, land_use_metadata_path, args.overwrite)


if __name__ == "__main__":
    main()

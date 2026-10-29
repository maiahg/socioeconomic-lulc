#!/usr/bin/env python3
"""Create a 2021 Ottawa population layer aligned to the 2020 land-use grid.

The first version uses uniform population density within each 2021
Dissemination Area. It writes:

* Idrisi/TerrSet .rst/.rdc versions of the same rasters;
* a GeoPackage containing the joined DA population attributes;
* a JSON validation summary.

If rasterio is installed, GeoTIFF copies are also written. The Idrisi files do
not require rasterio and are the primary outputs for TerrSet.

This is a baseline layer. It does not yet allocate population using buildings
or residential land-use weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import points
from shapely.strtree import STRtree


OTTAWA_DA_PREFIX = "3506"
TARGET_CRS = "EPSG:26918"  # NAD83 / UTM zone 18N
IDRISI_REFERENCE_SYSTEM = "utm-18n"
NODATA = -9999.0


def parse_args() -> argparse.Namespace:
    default_project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=default_project,
        help="Project root containing data/ and outputs/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs created by an earlier run.",
    )
    return parser.parse_args()


def read_idrisi_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        metadata[key.strip().lower()] = value.strip()
    return metadata


def create_population_profile(raw_census_path: Path, profile_path: Path) -> None:
    """Extract Ottawa DA populations from the raw Statistics Canada table."""

    print(f"Creating population profile: {profile_path}")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = profile_path.with_name(f".{profile_path.name}.tmp")
    extracted_rows = 0

    try:
        with raw_census_path.open(
            "r", encoding="cp1252", newline=""
        ) as source, temporary_path.open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            reader = csv.DictReader(source)
            required_columns = {
                "ALT_GEO_CODE",
                "GEO_LEVEL",
                "CHARACTERISTIC_NAME",
                "C1_COUNT_TOTAL",
            }
            missing_columns = required_columns.difference(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"Raw census file is missing columns: {sorted(missing_columns)}"
                )

            writer = csv.writer(destination)
            writer.writerow(["DAUID", "population_2021"])
            for row in reader:
                dauid = row["ALT_GEO_CODE"].strip()
                if not (
                    row["GEO_LEVEL"].strip() == "Dissemination area"
                    and dauid.startswith(OTTAWA_DA_PREFIX)
                    and row["CHARACTERISTIC_NAME"].strip() == "Population, 2021"
                ):
                    continue

                population = row["C1_COUNT_TOTAL"].strip().replace(",", "")
                if not population:
                    raise ValueError(f"Missing population for DAUID {dauid}.")
                writer.writerow([dauid, population])
                extracted_rows += 1

        if extracted_rows == 0:
            raise ValueError(
                f"No Ottawa dissemination-area population rows found in "
                f"{raw_census_path}."
            )
        os.replace(temporary_path, profile_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Created population profile with {extracted_rows:,} Ottawa DAs.")


def remove_if_allowed(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Re-run with --overwrite to replace it."
        )
    if path.exists():
        path.unlink()


def write_idrisi_real(
    array: np.ndarray,
    rst_path: Path,
    rdc_path: Path,
    *,
    width: int,
    height: int,
    resolution: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    title: str,
    display_min: float,
    display_max: float,
) -> None:
    """Write a float32 Idrisi Raster A.1 image and its documentation file."""

    values = np.asarray(array, dtype="<f4")
    if values.shape != (height, width):
        raise ValueError(f"Unexpected raster shape: {values.shape}")

    # Idrisi raster data are stored row by row from the upper-left cell.
    values.tofile(rst_path)
    rdc_text = "\n".join(
        [
            "file format : Idrisi Raster A.1",
            f"file title  : {title}",
            "data type   : real",
            "file type   : binary",
            f"columns     : {width}",
            f"rows        : {height}",
            f"ref. system : {IDRISI_REFERENCE_SYSTEM}",
            "ref. units  : m",
            "unit dist.  : 1",
            f"min. X      : {min_x:.7f}",
            f"max. X      : {max_x:.7f}",
            f"min. Y      : {min_y:.7f}",
            f"max. Y      : {max_y:.7f}",
            "pos'n error : unspecified",
            f"resolution  : {resolution:.7f}",
            f"min. value  : {display_min:.7f}",
            f"max. value  : {display_max:.7f}",
            f"display min : {display_min:.7f}",
            f"display max : {display_max:.7f}",
            "value units : unspecified",
            "value error : unspecified",
            f"flag value  : {NODATA:.1f}",
            "flag def'n  : missing data",
            "legend cats : 0",
            "lineage     :",
            "comment     : Created from Statistics Canada 2021 DA population data.",
            "",
        ]
    )
    rdc_path.write_text(rdc_text, encoding="utf-8")


def rasterize_polygons(
    geometries: np.ndarray,
    values: np.ndarray,
    *,
    width: int,
    height: int,
    resolution: float,
    min_x: float,
    max_y: float,
    fill: float,
    block_rows: int = 64,
) -> np.ndarray:
    """Rasterize polygon values using cell-centre assignment.

    This small dependency-free rasterizer uses Shapely's spatial index and
    reproduces the usual nearest-cell-centre behaviour used for polygon
    rasterization. Processing rows in blocks avoids creating all 6 million
    grid-cell points at once.
    """

    result = np.full((height, width), fill, dtype="float32")
    tree = STRtree(geometries)
    x_centres = min_x + (np.arange(width) + 0.5) * resolution
    y_centres = max_y - (np.arange(height) + 0.5) * resolution

    for row_start in range(0, height, block_rows):
        row_stop = min(row_start + block_rows, height)
        yy, xx = np.meshgrid(
            y_centres[row_start:row_stop], x_centres, indexing="ij"
        )
        cell_points = points(xx.ravel(), yy.ravel())
        matches = tree.query(cell_points, predicate="within")
        if matches.size:
            # STRtree returns query-geometry indices first and tree-geometry
            # indices second.
            point_indices = matches[0]
            polygon_indices = matches[1]
            block = result[row_start:row_stop].ravel()
            block[point_indices] = values[polygon_indices]

    return result


def write_optional_geotiff(
    path: Path,
    array: np.ndarray,
    *,
    width: int,
    height: int,
    resolution: float,
    min_x: float,
    max_y: float,
) -> bool:
    """Write a GeoTIFF when rasterio is available; otherwise skip it."""

    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        return False

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": from_origin(min_x, max_y, resolution, resolution),
        "nodata": NODATA,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)
    return True


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    data_dir = project_dir / "data"
    output_dir = project_dir / "outputs" / "ottawa_population"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_path = (
        project_dir
        / "outputs"
        / "ottawa_da_analysis"
        / "ottawa_da_census_2021_profile.csv"
    )
    raw_census_path = (
        data_dir
        / "census"
        / "2021"
        / "census"
        / "98-401-X2021006_English_CSV_data_Ontario.csv"
    )
    da_path = data_dir / "census" / "2021" / "da_boundaries" / "lda_000a21a_e.shp"
    land_use_metadata_path = data_dir / "LU Ottawa rst" / "LU2020 Ottawa_b1.rdc"

    for path in (da_path, land_use_metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    if not profile_path.exists():
        if not raw_census_path.exists():
            raise FileNotFoundError(raw_census_path)
        create_population_profile(raw_census_path, profile_path)

    density_rst = output_dir / "ottawa_population_density_2021.rst"
    density_rdc = output_dir / "ottawa_population_density_2021.rdc"
    count_rst = output_dir / "ottawa_population_count_2021.rst"
    count_rdc = output_dir / "ottawa_population_count_2021.rdc"
    da_gpkg = output_dir / "ottawa_population_2021_da.gpkg"
    summary_path = output_dir / "ottawa_population_2021_summary.json"

    density_tif = output_dir / "ottawa_population_density_2021.tif"
    count_tif = output_dir / "ottawa_population_count_2021.tif"

    output_paths = (
        density_rst,
        density_rdc,
        count_rst,
        count_rdc,
        da_gpkg,
        summary_path,
    )
    for path in output_paths:
        remove_if_allowed(path, args.overwrite)

    profile = pd.read_csv(profile_path, dtype={"DAUID": str})
    profile["DAUID"] = profile["DAUID"].str.strip()
    profile["population_2021"] = pd.to_numeric(
        profile["population_2021"], errors="coerce"
    )
    profile = profile[profile["DAUID"].str.startswith(OTTAWA_DA_PREFIX)].copy()

    if profile["DAUID"].duplicated().any():
        raise ValueError("The population table contains duplicate DAUID values.")
    if profile["population_2021"].isna().any():
        raise ValueError("Some Ottawa DAs have missing population values.")

    da_all = gpd.read_file(da_path)
    da_all["DAUID"] = da_all["DAUID"].astype(str).str.strip()
    da = da_all[da_all["DAUID"].str.startswith(OTTAWA_DA_PREFIX)].copy()
    da = da.merge(
        profile[["DAUID", "population_2021"]],
        on="DAUID",
        how="left",
        validate="one_to_one",
    )
    if da["population_2021"].isna().any():
        raise ValueError("Some DA boundary polygons did not receive population.")

    da = da.to_crs(TARGET_CRS)
    da["area_km2"] = da.geometry.area / 1_000_000
    da["population_density_km2"] = da["population_2021"] / da["area_km2"].replace(
        0, np.nan
    )

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
    density_values = da["population_density_km2"].to_numpy(dtype="float32")
    density_array = rasterize_polygons(
        geometries,
        density_values,
        width=width,
        height=height,
        resolution=resolution,
        min_x=min_x,
        max_y=max_y,
        fill=NODATA,
    )
    count_array = np.where(
        density_array == NODATA,
        NODATA,
        density_array * cell_area_km2,
    ).astype("float32")

    valid_count = count_array[count_array != NODATA]
    census_total = float(da["population_2021"].sum())
    raster_total = float(valid_count.sum())
    difference_pct = 100 * (raster_total - census_total) / census_total
    if abs(difference_pct) >= 2:
        raise ValueError(
            f"Raster population differs from Census total by {difference_pct:.3f}%. "
            "Check CRS, grid extent, and boundary coverage."
        )

    geotiff_written = (
        write_optional_geotiff(
            density_tif,
            density_array,
            width=width,
            height=height,
            resolution=resolution,
            min_x=min_x,
            max_y=max_y,
        )
        and write_optional_geotiff(
            count_tif,
            count_array,
            width=width,
            height=height,
            resolution=resolution,
            min_x=min_x,
            max_y=max_y,
        )
    )

    density_valid = density_array[density_array != NODATA]
    count_valid = count_array[count_array != NODATA]
    write_idrisi_real(
        density_array,
        density_rst,
        density_rdc,
        width=width,
        height=height,
        resolution=resolution,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        title="2021 Ottawa population density",
        display_min=float(density_valid.min()),
        display_max=float(density_valid.max()),
    )
    write_idrisi_real(
        count_array,
        count_rst,
        count_rdc,
        width=width,
        height=height,
        resolution=resolution,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        title="2021 Ottawa population per 30 m cell",
        display_min=float(count_valid.min()),
        display_max=float(count_valid.max()),
    )

    da.to_file(da_gpkg, layer="ottawa_population_2021_da", driver="GPKG")

    summary = {
        "census_year": 2021,
        "source_population_table": str(profile_path),
        "source_boundary_file": str(da_path),
        "land_use_grid_metadata": str(land_use_metadata_path),
        "target_crs": TARGET_CRS,
        "idrisi_reference_system": IDRISI_REFERENCE_SYSTEM,
        "width": width,
        "height": height,
        "resolution_m": resolution,
        "cell_area_km2": cell_area_km2,
        "ottawa_da_count": int(len(da)),
        "census_population_total": census_total,
        "raster_population_estimate": raster_total,
        "difference_pct": difference_pct,
        "allocation_method": "uniform population density within each Dissemination Area",
        "population_density_idrisi_rst": str(density_rst),
        "population_count_idrisi_rst": str(count_rst),
        "da_geopackage": str(da_gpkg),
    }
    if geotiff_written:
        summary["population_density_geotiff"] = str(density_tif)
        summary["population_count_geotiff"] = str(count_tif)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Ottawa DAs: {len(da):,}")
    print(f"Census population: {census_total:,.0f}")
    print(f"Raster population estimate: {raster_total:,.1f}")
    print(f"Difference: {difference_pct:.3f}%")
    print("\nCreated files:")
    for path in output_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()

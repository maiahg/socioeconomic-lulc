#!/usr/bin/env python3
"""Create Ottawa population layers aligned to the 2020 land-use grid.

The script uses uniform population density within each census year's
Dissemination Areas. It writes:

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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CensusYearSpec:
    year: int
    raw_census_path: Path
    boundary_path: Path
    encoding: str
    dauid_column: str
    characteristic_column: str
    characteristic_value: str
    population_column: str


CENSUS_SPECS = {
    2011: CensusYearSpec(
        year=2011,
        raw_census_path=Path(
            "data/census/2011/census/98-316-XWE2011001-1501-ONT.csv"
        ),
        boundary_path=Path("data/census/2011/da_boundaries/gda_000a11a_e.shp"),
        encoding="cp1252",
        dauid_column="Geo_Code",
        characteristic_column="Characteristic",
        characteristic_value="Population in 2011",
        population_column="Total",
    ),
    2016: CensusYearSpec(
        year=2016,
        raw_census_path=Path(
            "data/census/2016/census/98-401-X2016044_ONTARIO_English_CSV_data.csv"
        ),
        boundary_path=Path("data/census/2016/da_boundaries/lda_000a16a_e.shp"),
        encoding="utf-8-sig",
        dauid_column="GEO_CODE (POR)",
        characteristic_column=(
            "DIM: Profile of Dissemination Areas (2247)"
        ),
        characteristic_value="Population, 2016",
        population_column="Dim: Sex (3): Member ID: [1]: Total - Sex",
    ),
    2021: CensusYearSpec(
        year=2021,
        raw_census_path=Path(
            "data/census/2021/census/98-401-X2021006_English_CSV_data_Ontario.csv"
        ),
        boundary_path=Path("data/census/2021/da_boundaries/lda_000a21a_e.shp"),
        encoding="cp1252",
        dauid_column="ALT_GEO_CODE",
        characteristic_column="CHARACTERISTIC_NAME",
        characteristic_value="Population, 2021",
        population_column="C1_COUNT_TOTAL",
    ),
}


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
        "--years",
        type=int,
        nargs="+",
        default=list(CENSUS_SPECS),
        metavar="YEAR",
        help="Census years to generate (available: 2011, 2016, 2021).",
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


def create_population_profile(
    spec: CensusYearSpec,
    raw_census_path: Path,
    profile_path: Path,
) -> None:
    """Extract Ottawa DA populations from a raw Statistics Canada table."""

    print(f"Creating population profile: {profile_path}")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = profile_path.with_name(f".{profile_path.name}.tmp")
    extracted_rows = 0

    try:
        with raw_census_path.open(
            "r", encoding=spec.encoding, newline=""
        ) as source, temporary_path.open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            reader = csv.DictReader(source)
            required_columns = {
                spec.dauid_column,
                spec.characteristic_column,
                spec.population_column,
            }
            missing_columns = required_columns.difference(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"Raw census file is missing columns: {sorted(missing_columns)}"
                )

            writer = csv.writer(destination)
            writer.writerow(["DAUID", f"population_{spec.year}"])
            for row in reader:
                dauid = row[spec.dauid_column].strip()
                if not (
                    len(dauid) == 8
                    and dauid.startswith(OTTAWA_DA_PREFIX)
                    and row[spec.characteristic_column].strip()
                    == spec.characteristic_value
                ):
                    continue

                population = row[spec.population_column].strip().replace(",", "")
                if not population or population == "...":
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
            "comment     : Created from Statistics Canada DA population data.",
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


def generate_population_layer(
    spec: CensusYearSpec,
    project_dir: Path,
    land_use_metadata_path: Path,
    overwrite: bool,
) -> None:
    """Create the population layer for one census year."""

    profile_path = (
        project_dir
        / "outputs"
        / "ottawa_da_analysis"
        / f"ottawa_da_census_{spec.year}_profile.csv"
    )
    raw_census_path = project_dir / spec.raw_census_path
    da_path = project_dir / spec.boundary_path

    for path in (da_path, land_use_metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    if not profile_path.exists():
        if not raw_census_path.exists():
            raise FileNotFoundError(raw_census_path)
        create_population_profile(spec, raw_census_path, profile_path)

    output_dir = project_dir / "outputs" / "ottawa_population" / str(spec.year)
    output_dir.mkdir(parents=True, exist_ok=True)
    density_rst = output_dir / f"ottawa_population_density_{spec.year}.rst"
    density_rdc = output_dir / f"ottawa_population_density_{spec.year}.rdc"
    count_rst = output_dir / f"ottawa_population_count_{spec.year}.rst"
    count_rdc = output_dir / f"ottawa_population_count_{spec.year}.rdc"
    da_gpkg = output_dir / f"ottawa_population_{spec.year}_da.gpkg"
    summary_path = output_dir / f"ottawa_population_{spec.year}_summary.json"
    density_tif = output_dir / f"ottawa_population_density_{spec.year}.tif"
    count_tif = output_dir / f"ottawa_population_count_{spec.year}.tif"

    output_paths = (
        density_rst,
        density_rdc,
        count_rst,
        count_rdc,
        da_gpkg,
        summary_path,
    )
    for path in output_paths:
        remove_if_allowed(path, overwrite)

    population_column = f"population_{spec.year}"
    profile = pd.read_csv(profile_path, dtype={"DAUID": str})
    profile["DAUID"] = profile["DAUID"].str.strip()
    profile[population_column] = pd.to_numeric(
        profile[population_column], errors="coerce"
    )
    profile = profile[profile["DAUID"].str.startswith(OTTAWA_DA_PREFIX)].copy()

    if profile["DAUID"].duplicated().any():
        raise ValueError("The population table contains duplicate DAUID values.")
    if profile[population_column].isna().any():
        raise ValueError("Some Ottawa DAs have missing population values.")

    da_all = gpd.read_file(da_path)
    da_all["DAUID"] = da_all["DAUID"].astype(str).str.strip()
    da = da_all[da_all["DAUID"].str.startswith(OTTAWA_DA_PREFIX)].copy()
    da = da.merge(
        profile[["DAUID", population_column]],
        on="DAUID",
        how="left",
        validate="one_to_one",
    )
    if da[population_column].isna().any():
        raise ValueError("Some DA boundary polygons did not receive population.")

    da = da.to_crs(TARGET_CRS)
    da["area_km2"] = da.geometry.area / 1_000_000
    da["population_density_km2"] = da[population_column] / da["area_km2"].replace(
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
    census_total = float(da[population_column].sum())
    raster_total = float(valid_count.sum())
    difference_pct = 100 * (raster_total - census_total) / census_total
    if abs(difference_pct) >= 2:
        raise ValueError(
            f"Raster population differs from {spec.year} Census total by "
            f"{difference_pct:.3f}%. Check CRS, grid extent, and boundary coverage."
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
        title=f"{spec.year} Ottawa population density",
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
        title=f"{spec.year} Ottawa population per 30 m cell",
        display_min=float(count_valid.min()),
        display_max=float(count_valid.max()),
    )

    da.to_file(
        da_gpkg,
        layer=f"ottawa_population_{spec.year}_da",
        driver="GPKG",
    )

    summary = {
        "census_year": spec.year,
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

    print(f"Ottawa DAs ({spec.year}): {len(da):,}")
    print(f"Census population ({spec.year}): {census_total:,.0f}")
    print(f"Raster population estimate: {raster_total:,.1f}")
    print(f"Difference: {difference_pct:.3f}%")
    print("Created files:")
    for path in output_paths:
        print(f" - {path}")


def main() -> None:
    args = parse_args()
    requested_years = list(dict.fromkeys(args.years))
    unsupported_years = [year for year in requested_years if year not in CENSUS_SPECS]
    if unsupported_years:
        raise SystemExit(
            "Unsupported census year(s): "
            f"{', '.join(map(str, unsupported_years))}. "
            "The current 2001 and 2006 inputs are IVT files; export them to "
            "CSV before adding those years."
        )

    project_dir = args.project_dir.resolve()
    land_use_metadata_path = project_dir / "data" / "LU Ottawa rst" / "LU2020 Ottawa_b1.rdc"
    for year in requested_years:
        print(f"\nGenerating {year} population layer...")
        generate_population_layer(
            CENSUS_SPECS[year],
            project_dir,
            land_use_metadata_path,
            args.overwrite,
        )


if __name__ == "__main__":
    main()

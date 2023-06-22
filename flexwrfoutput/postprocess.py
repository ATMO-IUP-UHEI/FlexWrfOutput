"""
Additional functions needed in preprocess to secure compatibility to xWRF
"""
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from xwrf.postprocess import (
    _assign_coord_to_dim_of_different_name,
    _collapse_time_dim,
    _include_projection_coordinates,
    _make_units_pint_friendly,
    _modify_attrs_to_cf,
    _rename_dims,
)


def _prepare_conc_units(ds: xr.Dataset) -> xr.Dataset:
    """
    Change units of footprint data to be pint compatible.
    """
    ds.CONC.attrs["units"] = (
        "s m^3 kg^-1" if ds.CONC.units == "s m3 kg-1" else ds.CONC.units
    )
    return ds


def _make_attrs_consistent(ds: xr.Dataset) -> xr.Dataset:
    """
    Change attribute names and values of FLEXPART-WRF output to be compatible with xWRF
        postprocess.
    """

    assert (ds.south_north == list(range(ds.dims["south_north"]))).all()
    assert (ds.west_east == list(range(ds.dims["west_east"]))).all()
    y_center_index = (ds.dims["south_north"] - 1) / 2
    x_center_index = (ds.dims["west_east"] - 1) / 2

    ds.attrs["CEN_LAT"] = (
        ds.XLAT.interp(south_north=y_center_index, west_east=x_center_index)
        .squeeze()
        .values
    )
    ds.attrs["CEN_LON"] = (
        ds.XLONG.interp(south_north=y_center_index, west_east=x_center_index)
        .squeeze()
        .values
    )
    ds.attrs["MOAD_CEN_LAT"] = ds.attrs["CEN_LAT"]

    return ds


def _extract_simulation_start(ds: xr.Dataset) -> np.ndarray:
    """
    Extract simulation start from FLEXPART-WRF output.
    """
    unformatted_simulation_start = str(ds.SIMULATION_START_DATE) + str(
        ds.SIMULATION_START_TIME
    ).zfill(6)
    simulation_start = np.datetime64(
        datetime.strptime(unformatted_simulation_start, "%Y%m%d%H%M%S")
    )
    return simulation_start


def _assign_time_coord(ds: xr.Dataset) -> xr.Dataset:
    """
    Read native time format of FLEXPART-WRF and assign respective datetimes as
        coordinate.
    """
    formatted_times = pd.to_datetime(
        ds.Times.data.astype("str"), errors="raise", format="%Y%m%d_%H%M%S"
    )
    # Use center of averaging interval as dimension
    formatted_times += pd.Timedelta(ds.attrs["AVERAGING_TIME"], "seconds") / 2
    ds = ds.assign_coords(Time=("Time", formatted_times))
    # fmt: off
    ds.Time.attrs["description"] = (
        "Times of footprint output (center of averaging interval)"
    )
    # fmt: on
    return ds


def _split_releases_into_multiple_dimensions(ds: xr.Dataset) -> xr.Dataset:
    """Split releases according to the time and name of the release."""
    measurement_times = (
        _extract_simulation_start(ds)
        + ds.ReleaseTstart_end.values.mean(axis=1).astype("timedelta64[s]")
    ).astype("datetime64[ns]")

    measurement_names = ds.ReleaseName.values

    new_releases_coordinates = pd.MultiIndex.from_arrays(
        (measurement_times, measurement_names),
        names=("MTime", "MPlace"),
    )

    ds = ds.assign_coords(releases=new_releases_coordinates).unstack("releases")
    ds.MTime.attrs[
        "description"
    ] = "Times of measurement for each release (center of release interval)"
    ds.MPlace.attrs["description"] = "Names assigned to each release"
    return ds


def _add_measurement_information(ds: xr.Dataset) -> xr.Dataset:
    """Add information about measurement to dataset in additionional coordiantes for MTime
    and MPlace."""
    measurement_start = (
        _extract_simulation_start(ds)
        + ds.ReleaseTstart_end.isel(MPlace=0, ReleaseStartEnd=0).values
    ).astype("datetime64[ns]")
    measurement_end = (
        _extract_simulation_start(ds)
        + ds.ReleaseTstart_end.isel(MPlace=0, ReleaseStartEnd=1).values
    ).astype("datetime64[ns]")

    measurement_x_east = ds.ReleaseXstart_end.isel(MTime=0, ReleaseStartEnd=0).values
    measurement_x_center = (
        ds.ReleaseXstart_end.isel(MTime=0).mean("ReleaseStartEnd").values
    )
    measurement_x_west = ds.ReleaseXstart_end.isel(MTime=0, ReleaseStartEnd=1).values
    measurement_y_south = ds.ReleaseYstart_end.isel(MTime=0, ReleaseStartEnd=0).values
    measurement_y_center = (
        ds.ReleaseYstart_end.isel(MTime=0).mean("ReleaseStartEnd").values
    )
    measurement_y_north = ds.ReleaseYstart_end.isel(MTime=0, ReleaseStartEnd=1).values
    measurement_z_bottom = ds.ReleaseZstart_end.isel(MTime=0, ReleaseStartEnd=0).values
    measurement_z_center = (
        ds.ReleaseZstart_end.isel(MTime=0).mean("ReleaseStartEnd").values
    )
    measurement_z_top = ds.ReleaseZstart_end.isel(MTime=0, ReleaseStartEnd=1).values

    ds = ds.assign_coords(
        MTime_start=("MTime", measurement_start),
        MTime_end=("MTime", measurement_end),
        MPlace_x_east=("MPlace", measurement_x_east),
        MPlace_x_center=("MPlace", measurement_x_center),
        MPlace_x_west=("MPlace", measurement_x_west),
        MPlace_y_south=("MPlace", measurement_y_south),
        MPlace_y_center=("MPlace", measurement_y_center),
        MPlace_y_north=("MPlace", measurement_y_north),
        MPlace_z_bottom=("MPlace", measurement_z_bottom),
        MPlace_z_center=("MPlace", measurement_z_center),
        MPlace_z_top=("MPlace", measurement_z_top),
    )
    ds.MTime_start.attrs["description"] = "Start time of measurement for each release"
    ds.MTime_end.attrs["description"] = "End time of measurement for each release"
    ds.MPlace_x_east.attrs[
        "description"
    ] = "East boundary of measurement for each release"
    ds.MPlace_x_east.attrs["unit"] = "m"
    ds.MPlace_x_center.attrs["description"] = "Center of measurement for each release"
    ds.MPlace_x_center.attrs["unit"] = "m"
    ds.MPlace_x_west.attrs[
        "description"
    ] = "West boundary of measurement for each release"
    ds.MPlace_x_west.attrs["unit"] = "m"
    ds.MPlace_y_south.attrs[
        "description"
    ] = "South boundary of measurement for each release"
    ds.MPlace_y_south.attrs["unit"] = "m"
    ds.MPlace_y_center.attrs["description"] = "Center of measurement for each release"
    ds.MPlace_y_center.attrs["unit"] = "m"
    ds.MPlace_y_north.attrs[
        "description"
    ] = "North boundary of measurement for each release"
    ds.MPlace_y_north.attrs["unit"] = "m"
    ds.MPlace_z_bottom.attrs[
        "description"
    ] = "Bottom boundary of measurement for each release"
    ds.MPlace_z_bottom.attrs["unit"] = "m"
    ds.MPlace_z_center.attrs["description"] = "Center of measurement for each release"
    ds.MPlace_z_center.attrs["unit"] = "m"
    ds.MPlace_z_top.attrs[
        "description"
    ] = "Top boundary of measurement for each release"
    ds.MPlace_z_top.attrs["unit"] = "m"
    return ds


def _prepare_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """
    Set useful coordinates.
    """
    # if created with flexwrfinput z dim corresponds to z_stag of WRF
    ds = ds.rename_dims({"bottom_top": "z_stag"})
    ds = ds.assign_coords(z_stag=("z_stag", ds.ZTOP.values))
    ds.z_stag.attrs = dict(
        description="Hight at top of layer (above surface)", units="m"
    )
    # Set times as coordinates in datetime64 format
    ds = _assign_time_coord(ds)
    # take care of releases
    ds = _split_releases_into_multiple_dimensions(ds)
    ds = _add_measurement_information(ds)
    return ds


def _apply_xwrf_pipes(ds: xr.Dataset) -> xr.Dataset:
    ds = (
        ds.pipe(_modify_attrs_to_cf)
        .pipe(_make_units_pint_friendly)
        .pipe(_collapse_time_dim)
        .pipe(_assign_coord_to_dim_of_different_name)
        .pipe(_include_projection_coordinates)
        .pipe(_rename_dims)
    )
    return ds

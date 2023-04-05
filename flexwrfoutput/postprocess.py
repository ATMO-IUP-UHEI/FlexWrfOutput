"""
Additional functions needed in preprocess to secure compatibility to xWRF
"""
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


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

    ds.attrs["CEN_LAT"] = ds.XLAT.interp(
        south_north=y_center_index, west_east=x_center_index
    ).item()
    ds.attrs["CEN_LON"] = ds.XLONG.interp(
        south_north=y_center_index, west_east=x_center_index
    ).item()
    ds.attrs["MOAD_CEN_LAT"] = ds.attrs["CEN_LAT"]

    return ds


def _prepare_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """
    Set useful coordinates.
    """
    # if created with flexwrfinput z dim corresponds to z_stag of WRF
    ds = ds.rename_dims({"bottom_top": "bottom_top_stag"})
    ds = ds.assign_coords(z_height=("bottom_top_stag", ds.ZTOP.values))
    ds.z_height.attrs = dict(description="Top of layer (above surface)", units="m")
    # Set measurement times as coordinate for releases
    unformatted_simulation_start = str(ds.SIMULATION_START_DATE) + str(
        ds.SIMULATION_START_TIME
    ).zfill(6)
    simulation_start = np.datetime64(
        datetime.strptime(unformatted_simulation_start, "%Y%m%d%H%M%S")
    )
    measurement_times = simulation_start + ds.ReleaseTstart_end.values.mean(
        axis=1
    ).astype("timedelta64[s]")
    ds = ds.assign_coords(MTime=("releases", measurement_times))
    # fmt: off
    ds.MTime.attrs["description"] = (
        "Times of measurement for each release (center of release interval)"
    )
    # fmt: on
    # Set release name as coordinate
    ds = ds.assign_coords(releases_name=("releases", ds.ReleaseName.values))
    return ds


def _decode_times(ds: xr.Dataset) -> xr.Dataset:
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

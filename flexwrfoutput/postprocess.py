"""
Additional functions needed in preprocess to secure compatibility to xWRF
"""
from datetime import datetime

import numpy as np
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
    Change attribute names and values of FLEXPART-WRF output to be compatible with xWRF postprocess.
    """
    # CEN_LAT and CEN_LON are given for the mother domain not the inner domain
    ds.attrs["MOAD_CEN_LAT"] = ds.attrs["CEN_LAT"]

    ndims_lat = ds.dims["south_north"]
    ndims_lon = ds.dims["west_east"]

    lat_center_index = (
        int(ndims_lat // 2)
        if ndims_lat % 2 == 1
        else [int(ndims_lat // 2), int(ndims_lat // 2 - 1)]
    )
    lon_center_index = (
        int(ndims_lon // 2)
        if ndims_lon % 2 == 1
        else [int(ndims_lon // 2), int(ndims_lat // 2 - 1)]
    )

    ds.attrs["CEN_LAT"] = (
        ds.XLAT.isel(south_north=lat_center_index, west_east=lon_center_index)
        .mean()
        .values
    )
    ds.attrs["CEN_LON"] = (
        ds.XLONG.isel(south_north=lat_center_index, west_east=lon_center_index)
        .mean()
        .values
    )

    return ds


def _decode_times(ds: xr.Dataset) -> xr.Dataset:
    """
    Read native time format of FLEXPART-WRF and assings respective datetimes as coordinate.
    """
    # Set coordinates of output time
    unformatted_times = np.char.decode(ds.Times.values)
    formatted_times = np.array(
        [
            datetime.strptime(unformatted_time, "%Y%m%d_%H%M%S")
            for unformatted_time in unformatted_times
        ],
        dtype=np.datetime64,
    )
    # Use center of averaging interval as dimension
    formatted_times += np.timedelta64(ds.attrs["AVERAGING_TIME"], "s") / 2
    ds = ds.assign_coords(Time=("Time", formatted_times))
    ds.Time.attrs[
        "description"
    ] = "Times of footprint output (center of averaging interval)"

    # Set measurement times as coordinate for releases
    unformatted_simulation_start = str(ds.SIMULATION_START_DATE) + str(
        ds.SIMULATION_START_TIME
    )
    simulation_start = np.datetime64(
        datetime.strptime(unformatted_simulation_start, "%Y%m%d%H%M%S")
    )
    measurement_times = simulation_start + ds.ReleaseTstart_end.values.mean(
        axis=1
    ).astype("timedelta64[s]")
    ds = ds.assign_coords(MTime=("releases", measurement_times))
    ds.MTime.attrs[
        "description"
    ] = "Times of measurement for each release (center of release interval)"
    return ds

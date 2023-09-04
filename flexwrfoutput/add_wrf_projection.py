"""
Functions needed to only add the wrf projection to the dataset. Based on the
implementation in xWRF https://github.com/xarray-contrib/xwrf/blob/main/xwrf/grid.py#L18
"""

import pyproj
import xarray as xr


def _get_wrf_projection(ds: xr.Dataset) -> pyproj.CRS:
    # Use standards from a typical WRF file
    cen_lon = ds.CEN_LON
    proj_id = ds.MAP_PROJ

    pargs = {
        "x_0": 0,
        "y_0": 0,
        "a": 6370000,
        "b": 6370000,
        "lat_1": ds.TRUELAT1,
        "lat_2": getattr(ds, "TRUELAT2", ds.TRUELAT1),
        "lat_0": ds.MOAD_CEN_LAT,
        "lon_0": ds.STAND_LON,
        "center_lon": cen_lon,
    }

    if proj_id == 1:
        # Lambert
        pargs["proj"] = "lcc"
        del pargs["center_lon"]
    elif proj_id == 2:
        # Polar stereo
        pargs["proj"] = "stere"
        pargs["lat_ts"] = pargs["lat_1"]
        pargs["lat_0"] = 90.0
        del pargs["lat_1"], pargs["lat_2"], pargs["center_lon"]
    elif proj_id == 3:
        # Mercator
        pargs["proj"] = "merc"
        pargs["lat_ts"] = pargs["lat_1"]
        pargs["lon_0"] = pargs["center_lon"]
        del pargs["lat_0"], pargs["lat_1"], pargs["lat_2"], pargs["center_lon"]
    else:
        raise NotImplementedError(f"WRF proj not implemented yet: {proj_id}")

    # Construct the pyproj CRS (letting errors fail through)
    return pyproj.CRS(pargs)


def _add_wrf_projection(ds: xr.Dataset) -> xr.Dataset:
    """
    Add the wrf projection to the dataset.
    """
    wrf_crs = _get_wrf_projection(ds)
    ds["wrf_projection"] = (tuple(), wrf_crs, wrf_crs.to_cf())
    return ds

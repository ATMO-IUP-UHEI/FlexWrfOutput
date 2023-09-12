"""Define xarray accessors to make FLEXPART-WRF output more consistent to WRF data
    loaded via xWRF module."""
from __future__ import annotations  # noqa: F401

import xarray as xr

from flexwrfoutput.add_wrf_projection import _add_wrf_projection
from flexwrfoutput.postprocess import (
    _apply_xwrf_pipes,
    _make_attrs_consistent,
    _prepare_conc_units,
    _prepare_coordinates,
)


class FLEXWRFAccessor:
    """
    Common Dataset and DataArray accessor functionality.
    """

    def __init__(self, xarray_obj: xr.Dataset | xr.DataArray) -> FLEXWRFAccessor:
        self.xarray_obj = xarray_obj


@xr.register_dataset_accessor("flexwrf")
class FLEXWRFDatasetAccessor(FLEXWRFAccessor):
    """Adds a number of FLEXPART-WRF specific methods to xarray.Dataset objects."""

    def postprocess(self) -> xr.Dataset:
        ds = (
            self.xarray_obj.pipe(_prepare_conc_units)
            .pipe(_make_attrs_consistent)
            .pipe(_prepare_coordinates)
        )
        ds = _apply_xwrf_pipes(ds)
        return ds

    def add_wrf_projection(self) -> xr.Dataset:
        """
        Add the wrf projection to the dataset.
        """
        ds = self.xarray_obj.pipe(_add_wrf_projection)
        return ds

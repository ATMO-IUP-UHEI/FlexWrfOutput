from __future__ import annotations  # noqa: F401

import xarray as xr
from xwrf.postprocess import (
    _assign_coord_to_dim_of_different_name,
    _collapse_time_dim,
    _include_projection_coordinates,
    _make_units_pint_friendly,
    _modify_attrs_to_cf,
    _rename_dims,
)

from flexwrfoutput.postprocess import (
    _decode_times,
    _make_attrs_consistent,
    _make_dim_names_consistent,
    _prepare_conc_units,
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
            .pipe(_make_dim_names_consistent)
            .pipe(_decode_times)
            .pipe(_modify_attrs_to_cf)
            .pipe(_make_units_pint_friendly)
            .pipe(_collapse_time_dim)
            .pipe(_assign_coord_to_dim_of_different_name)
            .pipe(_include_projection_coordinates)
            .pipe(_rename_dims)
        )
        return ds
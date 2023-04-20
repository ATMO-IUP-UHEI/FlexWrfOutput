from pathlib import Path

import numpy as np
import pint
import pytest
import xarray as xr

from flexwrfoutput.openfiles import _combine_output_and_header
from flexwrfoutput.postprocess import (
    _assign_time_coord,
    _make_attrs_consistent,
    _prepare_conc_units,
    _prepare_coordinates,
)

FILE_EXAMPLES = Path(__file__).parent / "file_examples"


@pytest.fixture(scope="session")
def combined_flxout_ds(request):
    return _combine_output_and_header(
        xr.open_dataset(request.param[0]), xr.open_dataset(request.param[1])
    )


@pytest.mark.parametrize(
    "flxout_path, header_path",
    [
        (FILE_EXAMPLES / "flxout_degree.nc", FILE_EXAMPLES / "header_degree.nc"),
        (FILE_EXAMPLES / "flxout_meters.nc", FILE_EXAMPLES / "header_meters.nc"),
    ],
)
def test_prepare_conc_units(flxout_path, header_path):
    output = _combine_output_and_header(
        xr.open_dataset(flxout_path), xr.open_dataset(header_path)
    )
    fixed_output = _prepare_conc_units(output)
    # should fail if conversion is not correct
    pint.Unit(fixed_output.CONC.units)
    # test if CONC has units second
    output.CONC.attrs["units"] = "s"
    fixed_output = _prepare_conc_units(output)
    assert pint.Unit(fixed_output.CONC.units) == pint.Unit("s")


@pytest.mark.parametrize(
    "combined_flxout_ds",
    [
        (FILE_EXAMPLES / "flxout_degree.nc", FILE_EXAMPLES / "header_degree.nc"),
        (FILE_EXAMPLES / "flxout_meters.nc", FILE_EXAMPLES / "header_meters.nc"),
    ],
    indirect=True,
)
def test_make_attrs_consistent(combined_flxout_ds):
    fixed_output = _make_attrs_consistent(combined_flxout_ds)
    needed_variables = [
        "CEN_LON",
        "CEN_LAT",
        "TRUELAT1",
        "TRUELAT2",
        "MOAD_CEN_LAT",
        "MAP_PROJ",
        "STAND_LON",
    ]
    assert set(needed_variables).issubset(fixed_output.attrs.keys())


@pytest.mark.parametrize(
    "combined_flxout_ds",
    [
        (FILE_EXAMPLES / "flxout_degree.nc", FILE_EXAMPLES / "header_degree.nc"),
    ],
    indirect=True,
)
def test_assign_time_coord(combined_flxout_ds):
    fixed_output = _assign_time_coord(combined_flxout_ds)
    time_variable = "Time"
    assert time_variable in fixed_output.coords
    assert np.issubdtype(fixed_output[time_variable].values.dtype, np.datetime64)


@pytest.mark.parametrize(
    "combined_flxout_ds",
    [
        (FILE_EXAMPLES / "flxout_degree.nc", FILE_EXAMPLES / "header_degree.nc"),
    ],
    indirect=True,
)
def test_prepare_coordinates(combined_flxout_ds):
    combined_flxout_ds.attrs["SIMULATION_START_TIME"] = 0
    combined_flxout_ds = _prepare_coordinates(combined_flxout_ds)
    assert "bottom_top_stag" in combined_flxout_ds.dims
    assert "z_height" in combined_flxout_ds.coords
    assert "MTime" in combined_flxout_ds.coords
    assert "Time" in combined_flxout_ds.coords
    assert "releases_name" in combined_flxout_ds.coords

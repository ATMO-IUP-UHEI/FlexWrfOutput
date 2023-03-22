from pathlib import Path

import numpy as np
import pint
import pytest
import xarray as xr

from flexwrfoutput.openfiles import _combine_output_and_header
from flexwrfoutput.postprocess import (
    _decode_times,
    _make_attrs_consistent,
    _prepare_conc_units,
    _prepare_zdim,
)

FILE_EXAMPLES = Path(__file__).parent / "file_examples"


@pytest.fixture
def flxout_path_deg():
    return FILE_EXAMPLES / "flxout_degree.nc"


@pytest.fixture
def header_path_deg():
    return FILE_EXAMPLES / "header_degree.nc"


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
    "flxout_path, header_path",
    [
        (FILE_EXAMPLES / "flxout_degree.nc", FILE_EXAMPLES / "header_degree.nc"),
        (FILE_EXAMPLES / "flxout_meters.nc", FILE_EXAMPLES / "header_meters.nc"),
    ],
)
def test_make_attrs_consistent(flxout_path, header_path):
    output = _combine_output_and_header(
        xr.open_dataset(flxout_path), xr.open_dataset(header_path)
    )
    fixed_output = _make_attrs_consistent(output)
    needed_variables = [
        "CEN_LON",
        "CEN_LAT",
        "TRUELAT1",
        "TRUELAT2",
        "MOAD_CEN_LAT",
        "MAP_PROJ",
        "STAND_LON",
    ]
    for needed_variable in needed_variables:
        assert needed_variable in fixed_output.attrs.keys()


def test_decode_times_Times(flxout_path_deg, header_path_deg):
    output = _combine_output_and_header(
        xr.open_dataset(flxout_path_deg), xr.open_dataset(header_path_deg)
    )
    fixed_output = _decode_times(output)
    for time_variable in ["Time", "MTime"]:
        assert time_variable in fixed_output.coords
        assert np.issubdtype(fixed_output[time_variable].values.dtype, np.datetime64)


def test_decode_times_START_TIME(flxout_path_deg, header_path_deg):
    """Test special cases for SIMULATION_START_TIME"""
    output = _combine_output_and_header(
        xr.open_dataset(flxout_path_deg), xr.open_dataset(header_path_deg)
    )
    output.attrs["SIMULATION_START_TIME"] = 0
    # check if format is casted correctly and conversion doesn't raise error
    _decode_times(output)
    output.attrs["SIMULATION_START_TIME"] = 1004
    _decode_times(output)


def test_prepare_z_dim(flxout_path_deg, header_path_deg):
    output = _combine_output_and_header(
        xr.open_dataset(flxout_path_deg), xr.open_dataset(header_path_deg)
    )
    output = _prepare_zdim(output)
    assert "bottom_top_stag" in output.dims
    assert "z_height" in output.coords

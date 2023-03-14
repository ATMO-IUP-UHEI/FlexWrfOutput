from pathlib import Path

import numpy as np
import pint
import pytest
import xarray as xr

from flexwrfoutput.openfiles import combine
from flexwrfoutput.postprocess import (
    _decode_times,
    _make_attrs_consistent,
    _prepare_conc_units,
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
    output = combine(xr.open_dataset(flxout_path), xr.open_dataset(header_path))
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
    output = combine(xr.open_dataset(flxout_path), xr.open_dataset(header_path))
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


def test_decode_times(flxout_path_deg, header_path_deg):
    output = combine(xr.open_dataset(flxout_path_deg), xr.open_dataset(header_path_deg))
    fixed_output = _decode_times(output)
    assert "Time" in fixed_output.coords
    assert np.issubdtype(fixed_output.Time.values.dtype, np.datetime64)

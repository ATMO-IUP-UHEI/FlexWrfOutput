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
    _split_releases_into_multiple_dimensions,
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
        (
            FILE_EXAMPLES / "degree" / "flxout_degree.nc",
            FILE_EXAMPLES / "degree" / "header_degree.nc",
        ),
        (
            FILE_EXAMPLES / "meter" / "flxout_meters.nc",
            FILE_EXAMPLES / "meter" / "header_meters.nc",
        ),
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
        (
            FILE_EXAMPLES / "degree" / "flxout_degree.nc",
            FILE_EXAMPLES / "degree" / "header_degree.nc",
        ),
        (
            FILE_EXAMPLES / "meter" / "flxout_meters.nc",
            FILE_EXAMPLES / "meter" / "header_meters.nc",
        ),
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
        (
            FILE_EXAMPLES / "degree" / "flxout_degree.nc",
            FILE_EXAMPLES / "degree" / "header_degree.nc",
        ),
        (
            FILE_EXAMPLES / "meter" / "flxout_meters.nc",
            FILE_EXAMPLES / "meter" / "header_meters.nc",
        ),
    ],
    indirect=True,
)
def test_assign_time_coord(combined_flxout_ds):
    fixed_output = _assign_time_coord(combined_flxout_ds)
    time_variable = "Time"
    assert time_variable in fixed_output.coords
    assert np.issubdtype(fixed_output[time_variable].values.dtype, np.datetime64)
    assert fixed_output[time_variable][0].values == np.datetime64("2021-08-02T00:30:00")


@pytest.mark.parametrize(
    "combined_flxout_ds",
    [
        (
            FILE_EXAMPLES / "degree" / "flxout_degree.nc",
            FILE_EXAMPLES / "degree" / "header_degree.nc",
        ),
        (
            FILE_EXAMPLES / "meter" / "flxout_meters.nc",
            FILE_EXAMPLES / "meter" / "header_meters.nc",
        ),
    ],
    indirect=True,
)
def test_prepare_coordinates(combined_flxout_ds):
    combined_flxout_ds.attrs["SIMULATION_START_TIME"] = 0
    combined_flxout_ds = _prepare_coordinates(combined_flxout_ds)

    assert "z_stag" in combined_flxout_ds.sizes
    assert "MTime" in combined_flxout_ds.sizes
    assert "MPlace" in combined_flxout_ds.sizes
    assert "Time" in combined_flxout_ds.sizes

    assert "z_stag" in combined_flxout_ds.coords
    assert "MTime_start" in combined_flxout_ds.coords
    assert "MTime_end" in combined_flxout_ds.coords
    assert "MPlace_x_east" in combined_flxout_ds.coords
    assert "MPlace_x_center" in combined_flxout_ds.coords
    assert "MPlace_x_west" in combined_flxout_ds.coords
    assert "MPlace_y_south" in combined_flxout_ds.coords
    assert "MPlace_y_center" in combined_flxout_ds.coords
    assert "MPlace_y_north" in combined_flxout_ds.coords
    assert "MPlace_z_bottom" in combined_flxout_ds.coords
    assert "MPlace_z_center" in combined_flxout_ds.coords
    assert "MPlace_z_top" in combined_flxout_ds.coords


@pytest.mark.parametrize(
    "combined_flxout_ds",
    [
        (
            FILE_EXAMPLES / "degree" / "flxout_degree.nc",
            FILE_EXAMPLES / "degree" / "header_degree.nc",
        ),
        (
            FILE_EXAMPLES / "meter" / "flxout_meters.nc",
            FILE_EXAMPLES / "meter" / "header_meters.nc",
        ),
    ],
    indirect=True,
)
def test_split_releases_into_multiple_dimensions(combined_flxout_ds):
    ds = combined_flxout_ds.rename_dims({"bottom_top": "z_stag"})
    ds = ds.assign_coords(z_stag=("z_stag", ds.ZTOP.values))
    ds.z_stag.attrs = dict(
        description="Hight at top of layer (above surface)", units="m"
    )
    ds = _assign_time_coord(ds)
    ds = _split_releases_into_multiple_dimensions(ds)
    assert "MTime" in ds.sizes
    assert "MPlace" in ds.sizes

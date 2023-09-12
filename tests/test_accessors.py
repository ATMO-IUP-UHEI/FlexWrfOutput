from pathlib import Path

import pytest
import xarray as xr

import flexwrfoutput  # noqa: F401
from flexwrfoutput.openfiles import _combine_output_and_header

FILE_EXAMPLES = Path(__file__).parent / "file_examples"


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
def test_postprocess(flxout_path, header_path):
    output = _combine_output_and_header(
        xr.open_dataset(flxout_path), xr.open_dataset(header_path)
    )
    output.flexwrf.postprocess()
    assert output.CONC.chunks is None


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
def test_dask_postprocess(flxout_path, header_path):
    output = _combine_output_and_header(
        xr.open_dataset(flxout_path), xr.open_dataset(header_path)
    )
    output = output.chunk(
        dict(
            Time=1,
            south_north=1,
            west_east=1,
            bottom_top=1,
            releases=1,
        )
    )
    output.flexwrf.postprocess()
    assert output.CONC.chunks is not None


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
def test_add_wrf_projection(flxout_path, header_path):
    output = _combine_output_and_header(
        xr.open_dataset(flxout_path), xr.open_dataset(header_path)
    )
    output = output.flexwrf.postprocess()
    old_projection = output.wrf_projection.item()
    output = output.drop_vars("wrf_projection")
    output = output.flexwrf.add_wrf_projection()
    assert output.wrf_projection.item() == old_projection

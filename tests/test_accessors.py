from pathlib import Path

import pytest
import xarray as xr

import flexwrfoutput  # noqa: F401
from flexwrfoutput.openfiles import _combine_output_and_header

FILE_EXAMPLES = Path(__file__).parent / "file_examples"


@pytest.mark.parametrize(
    "flxout_path, header_path",
    [
        (FILE_EXAMPLES / "flxout_degree.nc", FILE_EXAMPLES / "header_degree.nc"),
        (FILE_EXAMPLES / "flxout_meters.nc", FILE_EXAMPLES / "header_meters.nc"),
    ],
)
def test_postprocess(flxout_path, header_path):
    output = _combine_output_and_header(
        xr.open_dataset(flxout_path), xr.open_dataset(header_path)
    )
    output.flexwrf.postprocess()

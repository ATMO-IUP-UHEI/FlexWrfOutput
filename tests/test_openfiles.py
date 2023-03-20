import numpy as np
import pytest
import xarray as xr

from flexwrfoutput.openfiles import (
    _combine_output_and_header,
    _get_output_paths,
    open_output,
)


@pytest.fixture
def flxout():
    times = np.array(
        [b"20210802_150000", b"20210802_140000", b"20210802_130000"], dtype="|S15"
    )

    flxout = xr.Dataset(
        data_vars=dict(
            Times=(["Time"], times),
            CONC=(
                [
                    "Time",
                    "ageclass",
                    "releases",
                    "bottom_top",
                    "south_north",
                    "west_east",
                ],
                np.ones((3, 1, 2, 2, 4, 5)),
            ),
        )
    )
    return flxout


@pytest.fixture
def header():
    header = xr.Dataset(
        data_vars=dict(
            XLONG_CORNER=(["south_north", "west_east"], np.ones((4, 5))),
            XLAT_CORNER=(["south_north", "west_east"], np.ones((4, 5))),
            ZTOP=(["bottom_top"], np.ones(2)),
            SPECIES=(["species"], np.ones(1)),
            AGECLASS=(["ageclass"], np.ones(1)),
            Times=(["Time"], []),
            ReleaseName=(["releases"], np.ones(2).astype(str)),
            ReleaseTstart_end=(["releases", "ReleaseStartEnd"], np.ones((2, 2))),
            ReleaseXstart_end=(["releases", "ReleaseStartEnd"], np.ones((2, 2))),
            ReleaseYstart_end=(["releases", "ReleaseStartEnd"], np.ones((2, 2))),
            ReleaseZstart_end=(["releases", "ReleaseStartEnd"], np.ones((2, 2))),
            ReleaseNP=(["releases"], np.ones(2)),
            ReleaseXMass=(["releases", "species"], np.ones((2, 1))),
            ReceptorLon=(["receptors"], []),
            ReceptorLat=(["receptors"], []),
            ReceptorName=(["receptors"], []),
            TOPOGRAPHY=(["south_north", "west_east"], np.ones((4, 5))),
            GRIDAREA=(["south_north", "west_east"], np.ones((4, 5))),
        ),
        coords=dict(
            XLONG=(["south_north", "west_east"], np.arange(4 * 5).reshape((4, 5))),
            XLAT=(["south_north", "west_east"], np.arange(4 * 5).reshape((4, 5))),
        ),
    )
    return header


@pytest.fixture
def output_directory(tmp_path, flxout, header):
    output_dir = tmp_path / "flexpart_output"
    output_dir.mkdir()
    flxout.to_netcdf(output_dir / "flxout.nc")
    header.to_netcdf(output_dir / "header.nc")
    return output_dir, [output_dir / "flxout.nc", output_dir / "header.nc"]


@pytest.fixture
def output_directory_empty(tmp_path):
    output_dir = tmp_path / "flexpart_output"
    output_dir.mkdir()
    filenames = ["flxout_test.nc", "header_test.nc", "other_file"]
    filepaths = [output_dir / file for file in filenames]
    [file.touch() for file in filepaths]
    return output_dir, filepaths


def test_combine(flxout, header):
    combination = _combine_output_and_header(flxout, header)
    assert "CONC" in combination.data_vars
    assert "XLONG" in combination.coords


def test_get_output_paths(output_directory_empty):
    output_dir, filepaths = output_directory_empty
    flxout_path, header_path = _get_output_paths(output_dir)
    assert flxout_path == filepaths[0]
    assert header_path == filepaths[1]


@pytest.mark.xfail
def test_fail_get_output_paths(output_directory_empty):
    output_dir, _ = output_directory_empty
    (output_dir / "flxout_test2.nc").touch()
    _get_output_paths(output_dir)


def test_open_output(output_directory, flxout, header):
    output_dir, filepaths = output_directory
    combination = open_output(output_dir)
    assert (combination.CONC == flxout.CONC).all()
    assert (combination.XLONG == header.XLONG).all()

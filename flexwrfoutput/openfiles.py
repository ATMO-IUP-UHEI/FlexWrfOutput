"""
Functions to handle raw netCDF output of FLEXPART-WRF.

Meant to be used before the postprocessing with the accessor.
"""
from pathlib import Path
from typing import Tuple, Union

import xarray as xr


def combine(flxout: xr.Dataset, header: xr.Dataset) -> xr.Dataset:
    """Combines dimensions of flxout with header to have full information of output in\
        one xarray.

    Args:
        flxout (xr.Dataset): Loaded flxout file.
        header (xr.Dataset): Loader header file.

    Returns:
        xr.Dataset: Combined Dataset
    """
    combined = xr.merge([flxout, header.drop_dims("Time")])
    return combined


def get_output_paths(path: Union[str, Path]) -> Tuple[Path, Path]:
    """Finds header and flxout files in directory and returns their paths.

    Args:
        path (Union[str, Path]): Path of output directory of FLEXPART-WRF.

    Returns:
        Tuple[Path, Path]: (flxout path, header path)
    """
    path = Path(path)
    header_files = [file for file in path.iterdir() if "header" in str(file)]
    flxout_files = [file for file in path.iterdir() if "flxout" in str(file)]

    assert (
        len(header_files) == 1
    ), f"Didn't find unique header files in {path}: {header_files}"
    assert (
        len(flxout_files) == 1
    ), f"Didn't find unique flxout file in {path}: {flxout_files}"

    return flxout_files[0], header_files[0]


def open_output(output_dir: Union[str, Path]) -> xr.Dataset:
    """Finds output of FLEXPART-WRF in a directory and merges header and footprint data.

    Args:
        output_dir (Union[str, Path]): Directory with FLEXPART-WRF output files.

    Returns:
        xr.Dataset: Merged data.
    """
    output_dir = Path(output_dir)
    flxout_path, header_path = get_output_paths(output_dir)
    flxout = xr.open_dataset(flxout_path)
    header = xr.open_dataset(header_path)
    output = combine(flxout, header)
    return output

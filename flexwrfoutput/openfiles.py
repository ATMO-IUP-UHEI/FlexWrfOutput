"""
Functions to handle raw netCDF output of FLEXPART-WRF.

Meant to be used before the postprocessing with the accessor.
"""
from pathlib import Path
from typing import Tuple, Union

import xarray as xr


def _combine_output_and_header(flxout: xr.Dataset, header: xr.Dataset) -> xr.Dataset:
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


class AmbiguousPathError(Exception):
    pass


def _get_output_paths(path: Union[str, Path]) -> Tuple[Path, Path]:
    """Finds header and flxout files in directory and returns their paths.

    Args:
        path (Union[str, Path]): Path of output directory of FLEXPART-WRF.

    Returns:
        Tuple[Path, Path]: (flxout path, header path)
    """
    path = Path(path)

    header_files = list(path.glob("header*"))
    flxout_files = list(path.glob("flxout*"))

    if len(header_files) == 0:
        raise (
            FileNotFoundError(f"Did not find a header file in given directory {path}")
        )
    if len(flxout_files) == 0:
        raise (
            FileNotFoundError(f"Did not find a flxout file in given directory {path}")
        )
    if len(header_files) > 1:
        raise (
            AmbiguousPathError(f"Found multiple header files in given directory {path}")
        )
    if len(flxout_files) > 1:
        raise (
            AmbiguousPathError(f"Found multiple flxout files in given directory {path}")
        )

    return flxout_files[0], header_files[0]


def open_output(output_dir: Union[str, Path]) -> xr.Dataset:
    """Finds output of FLEXPART-WRF in a directory and merges header and footprint data.

    Args:
        output_dir (Union[str, Path]): Directory with FLEXPART-WRF output files.

    Returns:
        xr.Dataset: Merged data.
    """
    output_dir = Path(output_dir)
    flxout_path, header_path = _get_output_paths(output_dir)
    output = _combine_output_and_header(
        xr.open_dataset(flxout_path), xr.open_dataset(header_path)
    )
    return output

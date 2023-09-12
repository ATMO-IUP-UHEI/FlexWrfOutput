"""
Functions to handle raw netCDF output of FLEXPART-WRF.

Meant to be used before the postprocessing with the accessor.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

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

    if not (len(header_files) or len(flxout_files)):
        missing_file = " or ".join(
            [
                fname
                for fname, file_exists in [
                    ("header", len(header_files)),
                    ("flxout", len(flxout_files)),
                ]
                if not file_exists
            ]
        )
        raise (
            FileNotFoundError(
                f"Did not find a {missing_file} file in given directory {path}"
            )
        )
    elif len(header_files) > 1 or len(flxout_files) > 1:
        duplicate_filetype = " and ".join(
            [
                fname
                for fname, num_files in [
                    ("header", len(header_files)),
                    ("flxout", len(flxout_files)),
                ]
                if num_files > 1
            ]
        )
        raise (
            AmbiguousPathError(
                f"Found multiple {duplicate_filetype} files in given directory {path}"
            )
        )

    return flxout_files[0], header_files[0]


def open_output(
    output_dir: Union[str, Path],
    flxout_chunks: Optional[dict] = None,
    header_chunks: Optional[dict] = None,
) -> xr.Dataset:
    """Finds output of FLEXPART-WRF in a directory and merges header and footprint data.

    Args:
        output_dir (Union[str, Path]): Directory with FLEXPART-WRF output files.

    Returns:
        xr.Dataset: Merged data.
    """
    flxout_path, header_path = _get_output_paths(Path(output_dir))
    return _combine_output_and_header(
        xr.open_dataset(flxout_path, chunks=flxout_chunks),
        xr.open_dataset(header_path, chunks=header_chunks),
    )

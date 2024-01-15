# FlexWrfOutput

An extension to [xWRF](https://github.com/xarray-contrib/xwrf) to consistently handle output of [FLEXPART-WRF](https://www.flexpart.eu/wiki/FpLimitedareaWrf).

## Installation
### Requirements:
At the moment the installation is **only works for Linux systems**. There are acouple of requirements for the installation of the module. To initialize a conda environment that fulfills the requirements you can install it from the file `ci/environment.yml` if you have `poetry` already installed or `ci/environment_ci.yml` if not.
```bash
conda env create -f ci/environment[_ci].yml
```
The required software is:
 * `python3.10`
 * `netcdf4`
 * `poetry`

### Download
You need to download the module from the git repository. You can do this with the following command:
```bash
git clone git@github.com:ATMO-IUP-UHEI/FlexWrfOutput.git [--branch <branch>]
```
where `<branch>` is the name of the branch you want to download. If you don't specify the branch, the `main` branch will be downloaded.

### Installation
To install the module you can use `poetry` or `pip` from the main folder of the directory. The installation with `poetry` is recommended. To install the module with `poetry` you can use the following command:
```bash
poetry install
```
To install the module with `pip` you can use the following command:
```bash
pip install .
```

### Testing
Tests are written in `pytest`. To run the tests in your environment you can use the following command:
```bash
python -m pytest
```

## Usage
The tools presented by `FlexWrfOutput` meant to load and postprocess the output of `FLEXPART-WRF`.
### Load output data
To load data import the `open_output` function and provide it with the output directory (only works for non nested output).
```python
from flexwrfoutput import open_output

ds = open_output("/path/to/output_directory")
```

### Postprocess output data
There are two applications to postprocess the output data. The goal is to end up in a format that is compatible with `WRF` output that is postprocessed with [xWRF](https://github.com/xarray-contrib/xwrf). This is performed by the `postprocess` `xarray`-accessor:
```python
# Registers the accessor
import flexwrfoutput as fwo

# Load and postprocess data
ds = fwo.open_output("/path/to/output_directory").flexwrf.postprocess()
```
This step additionally adds the projection of the data, which cannot be simply saved into a NetCDF-format. However you cas save the data and add the projection after the loading of the data with the `add_wrf_projection` accessor:
```python
# Registers the accessor
import flexwrfoutput as fwo
import xarray as xr

# Load data
ds = xr.open_dataset("/path/to/postprocessed_data.nc").flexwrf.add_wrf_projection()
```

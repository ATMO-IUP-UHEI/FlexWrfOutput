from pathlib import Path

import pytest

import flexwrfoutput as fwo

FILE_EXAMPLES = Path(__file__).parent / "file_examples"


@pytest.fixture(
    params=[
        (FILE_EXAMPLES / "degree"),
        (FILE_EXAMPLES / "meter"),
    ]
)
def output_directory(request):
    return request.param


def test_open_and_postprocess(output_directory):
    output = fwo.open_output(output_directory)
    output = output.flexwrf.postprocess()
    assert output.CONC.chunks is None
    assert (output.MTime.values == output.MTime_start.values).all()
    assert (output.MTime.values == output.MTime_end.values).all()


def test_dask_open_and_postprocess(output_directory):
    output = fwo.open_output(
        output_directory, flxout_chunks=dict(Time=1), header_chunks=dict(releases=1)
    )
    output = output.flexwrf.postprocess()
    assert output.CONC.chunks is not None
    assert (output.MTime.values == output.MTime_start.values).all()
    assert (output.MTime.values == output.MTime_end.values).all()

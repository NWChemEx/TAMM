# tests/test_print_utils.py
from contextlib import contextmanager
from pathlib import Path
import re

import numpy as np
import pytest

pytamm = pytest.importorskip("pytamm")


@pytest.fixture(scope="session")
def tamm_ec():
    """Initialize TAMM once for the test session."""
    pytamm.initialize(["pytest"], False)

    pg = pytamm.ProcGroup.create_world_coll()
    ec = pytamm.ExecutionContext(
        pg,
        pytamm.DistributionKind.nw,
        pytamm.MemoryManagerKind.ga,
    )

    yield ec

    try:
        ec.flush_and_sync()
    finally:
        pytamm.finalize()


def rank(ec) -> int:
    return int(ec.pg_rank())


def txt_path(stem: Path) -> Path:
    """TAMM print_tensor/print_tensor_all/print_dense_tensor append '.txt'."""
    return Path(str(stem) + ".txt")


@contextmanager
def tamm_tensor(ec, array, tilesize=2):
    t = pytamm.from_numpy(np.asarray(array), ec, tilesize=tilesize)
    try:
        yield t
    finally:
        try:
            if t.is_allocated():
                t.deallocate()
        except Exception:
            pass


def make_data(dtype):
    arr = np.array(
        [
            [0.0, 1.25],
            [2.5, -3.0],
        ],
        dtype=dtype,
    )
    if np.issubdtype(dtype, np.complexfloating):
        arr[0, 0] = 1.0 + 2.0j
        arr[0, 1] = 1.25 + 0.5j
        arr[1, 0] = 2.5 - 1.0j
        arr[1, 1] = -3.0 + 0.25j
    return arr


def test_print_varlist(tamm_ec, capfd):
    capfd.readouterr()

    pytamm.print_varlist("alpha", 123, "omega")

    out = capfd.readouterr().out
    assert "alpha,123,omega" in out


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_print_tensor_and_print_tensor_all_to_files(tamm_ec, tmp_path, dtype):
    r = rank(tamm_ec)

    with tamm_tensor(tamm_ec, make_data(dtype)) as tensor:
        stem = tmp_path / f"print_tensor_{dtype.__name__}_rank{r}"
        pytamm.print_tensor(tensor, str(stem))

        text = txt_path(stem).read_text()
        assert "tensor dims" in text
        assert "actual tensor size" in text
        assert "blockid:" in text

        stem_all = tmp_path / f"print_tensor_all_{dtype.__name__}_rank{r}"
        pytamm.print_tensor_all(tensor, str(stem_all))

        text_all = txt_path(stem_all).read_text()
        assert "tensor dims" in text_all
        assert "actual tensor size" in text_all
        assert "blockid:" in text_all

        labeled = tensor()

        stem_labeled = tmp_path / f"print_labeled_tensor_file_{dtype.__name__}_rank{r}"
        pytamm.print_tensor(labeled, str(stem_labeled))

        text_labeled = txt_path(stem_labeled).read_text()
        assert "tensor dims" in text_labeled
        assert "actual tensor size" in text_labeled
        assert "blockid:" in text_labeled

        stem_labeled_all = tmp_path / f"print_labeled_tensor_all_file_{dtype.__name__}_rank{r}"
        pytamm.print_tensor_all(labeled, str(stem_labeled_all))

        text_labeled_all = txt_path(stem_labeled_all).read_text()
        assert "tensor dims" in text_labeled_all
        assert "actual tensor size" in text_labeled_all
        assert "blockid:" in text_labeled_all


def test_print_labeled_tensor_and_print_tensor_reshaped_stdout(tamm_ec, capfd):
    with tamm_tensor(tamm_ec, make_data(np.float64)) as tensor:
        labeled = tensor()

        capfd.readouterr()
        pytamm.print_labeled_tensor(labeled)
        # print_varlist uses std::endl, which flushes C++ std::cout.
        pytamm.print_varlist("__flush_after_print_labeled_tensor__")

        out = capfd.readouterr().out
        assert "tensor dims" in out
        assert "actual tensor size" in out
        assert "__flush_after_print_labeled_tensor__" in out

        capfd.readouterr()
        pytamm.print_tensor_reshaped(labeled, labeled.labels())
        pytamm.print_varlist("__flush_after_print_tensor_reshaped__")

        out = capfd.readouterr().out
        assert "tensor dims" in out
        assert "actual tensor size" in out
        assert "__flush_after_print_tensor_reshaped__" in out


def test_print_vector_real_and_complex(tamm_ec, tmp_path):
    r = rank(tamm_ec)

    real_file = tmp_path / f"print_vector_real_rank{r}.txt"
    pytamm.print_vector([1.0, -2.5, 3.0], str(real_file))

    real_text = real_file.read_text()
    assert "1\t1.000000000000" in real_text
    assert "2\t-2.500000000000" in real_text
    assert "3\t3.000000000000" in real_text

    complex_file = tmp_path / f"print_vector_complex_rank{r}.txt"
    pytamm.print_vector([1.0 + 2.0j, -3.0 + 0.5j], str(complex_file))

    complex_text = complex_file.read_text()
    assert "1\t" in complex_text
    assert "2\t" in complex_text
    assert "(" in complex_text
    assert "," in complex_text


def test_print_max_above_threshold(tamm_ec, tmp_path):
    r = rank(tamm_ec)

    with tamm_tensor(tamm_ec, make_data(np.float64)) as tensor:
        out_file = tmp_path / f"print_max_above_threshold_rank{r}.txt"

        pytamm.print_max_above_threshold(tensor, 2.0, str(out_file))

        text = out_file.read_text()
        assert "2.500000000000" in text
        assert "-3.000000000000" in text
        assert "1.250000000000" not in text


def test_print_dense_tensor_to_file_and_predicate(tamm_ec, tmp_path):
    r = rank(tamm_ec)

    with tamm_tensor(tamm_ec, make_data(np.float64)) as tensor:
        dense_stem = tmp_path / f"print_dense_tensor_rank{r}"

        pytamm.print_dense_tensor(tensor, str(dense_stem))

        # print_dense_tensor writes only on rank 0.
        if r == 0:
            dense_file = txt_path(dense_stem)
            text = dense_file.read_text()

            assert re.search(r"\b1\s+2\s+1\.2500000000", text)
            assert re.search(r"\b2\s+1\s+2\.5000000000", text)
            assert re.search(r"\b2\s+2\s+-3\.0000000000", text)

            first_size = dense_file.stat().st_size

            pytamm.print_dense_tensor(tensor, str(dense_stem), True)

            assert dense_file.stat().st_size > first_size
        else:
            pytamm.print_dense_tensor(tensor, str(dense_stem), True)

        pred_stem = tmp_path / f"print_dense_tensor_predicate_rank{r}"

        pytamm.print_dense_tensor(
            tensor,
            lambda coord: coord[0] == 0,
            str(pred_stem),
        )

        if r == 0:
            pred_text = txt_path(pred_stem).read_text()

            assert re.search(r"\b1\s+2\s+1\.2500000000", pred_text)
            assert not re.search(r"\b2\s+1\s+2\.5000000000", pred_text)
            assert not re.search(r"\b2\s+2\s+-3\.0000000000", pred_text)


def test_print_memory_usage_helpers(tamm_ec, capfd):
    r = rank(tamm_ec)

    capfd.readouterr()

    pytamm.print_memory_usage(r, "generic-memory")
    pytamm.print_memory_usage_double(r, "double-memory")
    pytamm.print_memory_usage_complex_double(r, "complex-memory")

    out = capfd.readouterr().out

    if r == 0:
        assert "generic-memory" in out
        assert "double-memory" in out
        assert "complex-memory" in out
        assert "allocation count" in out
        assert "deallocation count" in out
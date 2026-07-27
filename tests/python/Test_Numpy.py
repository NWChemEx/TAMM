# Test_Numpy.py
import sys
import pytest
import numpy as np
import pytamm as tamm


@pytest.fixture(scope="session", autouse=True)
def tamm_runtime():
    tamm.initialize(["pytest", "Test_Numpy.py"], False)
    yield
    tamm.finalize()


@pytest.fixture(scope="module")
def ec():
    pg = tamm.ProcGroup.create_world_coll()
    return tamm.ExecutionContext(pg, tamm.DistributionKind.nw, tamm.MemoryManagerKind.ga)


def make_tis(n, tiles):
    assert sum(tiles) == n
    return tamm.TiledIndexSpace(tamm.IndexSpace(tamm.range(n)), list(tiles))


def dense_shape(tensor):
    return tuple(int(tis.max_num_indices()) for tis in tensor.tiled_index_spaces())


# Deterministic function used to generate values for filling tensors and arrays
def value_at(idx, complex_values=False):
    real = 1.25
    for axis, coord in enumerate(idx):
        real += float(coord + 1) * float(10 ** axis)

    if complex_values:
        imag = -0.5
        for axis, coord in enumerate(idx):
            imag -= float(axis + 2) * float(coord + 1)
        return complex(real, imag)

    return real


def fill_distributed_tensor(tensor, complex_values=False, block_predicate=None):
    """
    Fill a distributed TAMM tensor with deterministic values and build the
    matching dense NumPy reference.

    If block_predicate is provided, blocks for which it returns False are
    explicitly filled with zeros. This avoids relying on TensorInfo/nonzero
    pruning behavior, which may still expose all block IDs in loop_nest().

    Used to create theoretically equivalent NumPy and TAMM tensors.
    """
    shape = dense_shape(tensor)
    dtype = np.complex128 if complex_values else np.float64
    expected = np.zeros(shape, dtype=dtype)

    def fill_block(blockid, buf):
        blockid = tuple(int(x) for x in blockid)
        block_dims = tuple(int(x) for x in tensor.block_dims(blockid))
        block_offsets = tuple(int(x) for x in tensor.block_offsets(blockid))

        assert int(np.prod(block_dims, dtype=np.int64)) == int(buf.size)

        if block_predicate is not None and not block_predicate(blockid):
            buf[:] = 0
            return

        # For each index in the buffer, assign a deterministic value computed in value_at()
        for flat_i in range(int(buf.size)):
            local_idx = np.unravel_index(flat_i, block_dims, order="C")
            global_idx = tuple(block_offsets[i] + local_idx[i] for i in range(len(block_dims)))
            val = value_at(global_idx, complex_values=complex_values)
            buf[flat_i] = val
            expected[global_idx] = val

    tamm.update_tensor(tensor(), fill_block)
    return expected


def fill_local_tensor(local_tensor, complex_values=False):
    shape = tuple(int(x) for x in local_tensor.dim_sizes())
    dtype = np.complex128 if complex_values else np.float64

    expected = np.empty(shape, dtype=dtype)
    # For each index in the array, assign a deterministic value computed in value_at()
    for idx in np.ndindex(shape):
        expected[idx] = value_at(idx, complex_values=complex_values)

    buf = local_tensor.access_local_buf()
    assert buf is not None
    np.asarray(buf)[:] = expected.reshape(-1, order="C")
    return expected


# Function to verify an array and a tensor are equivalent
def assert_array_matches(arr, expected, dtype=None):
    expected = np.asarray(expected)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == expected.shape
    assert arr.flags.c_contiguous

    if dtype is None:
        assert arr.dtype == expected.dtype
        expected_cast = expected
    else:
        assert arr.dtype == np.dtype(dtype)
        expected_cast = expected.astype(dtype)

    if arr.dtype in (np.dtype(np.float32), np.dtype(np.complex64)):
        np.testing.assert_allclose(arr, expected_cast, rtol=1.0e-6, atol=1.0e-6)
    else:
        np.testing.assert_allclose(arr, expected_cast, rtol=1.0e-12, atol=1.0e-12)


# Tests various conversions from TAMM to NumPy
def assert_numpy_entry_points(obj, expected, cast_dtype):
    # Method API
    assert_array_matches(obj.to_numpy(), expected)
    assert_array_matches(obj.to_numpy(cast_dtype), expected, dtype=cast_dtype)

    # Module-level free-function API
    assert_array_matches(tamm.to_numpy(obj), expected)
    assert_array_matches(tamm.to_numpy(obj, dtype=cast_dtype), expected, dtype=cast_dtype)

    # NumPy array protocol API
    assert_array_matches(np.asarray(obj), expected)
    assert_array_matches(np.asarray(obj, dtype=cast_dtype), expected, dtype=cast_dtype)

    # Direct __array__ calls exercise dtype/copy arguments explicitly.
    assert_array_matches(obj.__array__(None, False), expected)
    assert_array_matches(obj.__array__(np.dtype(cast_dtype), True), expected, dtype=cast_dtype)

    # to_numpy/__array__ should return an independent NumPy allocation, not a
    # mutable view into TAMM storage.
    arr = obj.to_numpy()
    if arr.size:
        original = arr.flat[0]
        arr.flat[0] = original + (123.0 + 456.0j if np.iscomplexobj(arr) else 123.0)
        assert_array_matches(obj.to_numpy(), expected)


def deallocate_if_needed(*objs):
    for obj in objs:
        if obj is not None and obj.is_allocated():
            obj.deallocate()



def test_distributed_dense_double_rank3_to_numpy_all_entry_points(ec):
    """
    High level test that builds rank-3 tensors and tests conversion to NumPy
    """
    spaces = [
        make_tis(5, [2, 3]),
        make_tis(4, [1, 3]),
        make_tis(3, [2, 1]),
    ]

    tensor = tamm.TensorDouble(spaces)
    tensor.allocate_self(ec)

    try:
        expected = fill_distributed_tensor(tensor, complex_values=False)
        ec.flush_and_sync()

        assert expected.shape == (5, 4, 3)

        assert_numpy_entry_points(tensor, expected, np.float32)
        assert_numpy_entry_points(tensor(), expected, np.float32)
    finally:
        deallocate_if_needed(tensor)


def test_distributed_double_to_numpy_preserves_zero_tile_blocks(ec):
    """
    Tests that the NumPy conversion preserves zeros in untargeted blocks.

    This intentionally does not rely on TensorInfo/nonzero pruning because this
    binding/runtime path may still expose all blocks through loop_nest().
    """
    tis = make_tis(5, [2, 3])

    tensor = tamm.TensorDouble([tis, tis])
    tensor.allocate_self(ec)

    try:
        expected = fill_distributed_tensor(
            tensor,
            complex_values=False,
            block_predicate=lambda blockid: int(blockid[0]) == int(blockid[1]),
        )
        ec.flush_and_sync()

        assert expected.shape == (5, 5)
        assert np.count_nonzero(expected) < expected.size
        assert np.any(expected == 0.0)

        assert_numpy_entry_points(tensor, expected, np.float32)
        assert_numpy_entry_points(tensor(), expected, np.float32)
    finally:
        deallocate_if_needed(tensor)


def test_distributed_complex_rank5_to_numpy_all_entry_points(ec):
    """
    High level test that builds complex-valued rank-5 tensors and tests conversion to NumPy
    """
    spaces = [
        make_tis(2, [1, 1]),
        make_tis(3, [1, 2]),
        make_tis(2, [2]),
        make_tis(2, [1, 1]),
        make_tis(2, [1, 1]),
    ]

    tensor = tamm.TensorComplexDouble(spaces)
    tensor.allocate_self(ec)

    try:
        expected = fill_distributed_tensor(tensor, complex_values=True)
        ec.flush_and_sync()

        assert expected.shape == (2, 3, 2, 2, 2)

        assert_numpy_entry_points(tensor, expected, np.complex64)
        assert_numpy_entry_points(tensor(), expected, np.complex64)
    finally:
        deallocate_if_needed(tensor)


def test_local_double_to_numpy_all_entry_points(ec):
    """
    Tests conversion of a LocalTensor of rank 3 to a NumPy array
    """
    spaces = [
        make_tis(3, [1, 2]),
        make_tis(4, [2, 2]),
        make_tis(2, [1, 1]),
    ]

    local = tamm.LocalTensorDouble(spaces)
    local.allocate_self(ec)

    try:
        expected = fill_local_tensor(local, complex_values=False)

        assert expected.shape == tuple(int(x) for x in local.dim_sizes())

        assert_numpy_entry_points(local, expected, np.float32)
        assert_numpy_entry_points(local(), expected, np.float32)
    finally:
        deallocate_if_needed(local)


def test_local_complex_to_numpy_all_entry_points(ec):
    """
    Tests conversion of a complex-valued LocalTensor of rank 3 to a NumPy array
    """
    spaces = [
        make_tis(2, [1, 1]),
        make_tis(3, [1, 2]),
    ]

    local = tamm.LocalTensorComplexDouble(spaces)
    local.allocate_self(ec)

    try:
        expected = fill_local_tensor(local, complex_values=True)

        assert expected.shape == tuple(int(x) for x in local.dim_sizes())

        assert_numpy_entry_points(local, expected, np.complex64)
        assert_numpy_entry_points(local(), expected, np.complex64)
    finally:
        deallocate_if_needed(local)


def test_to_numpy_rejects_unallocated_distributed_and_local_tensors():
    """
    Test to ensure NumPy raises an error if an unallocated tensor is attempted to be converted to an array
    """
    tis = make_tis(4, [2, 2])

    dist = tamm.TensorDouble([tis])
    dist_labeled = dist()

    with pytest.raises(ValueError, match="unallocated"):
        dist.to_numpy()
    with pytest.raises(ValueError, match="unallocated"):
        tamm.to_numpy(dist)
    with pytest.raises(ValueError, match="unallocated"):
        np.asarray(dist)

    with pytest.raises(ValueError, match="unallocated"):
        dist_labeled.to_numpy()
    with pytest.raises(ValueError, match="unallocated"):
        tamm.to_numpy(dist_labeled)
    with pytest.raises(ValueError, match="unallocated"):
        np.asarray(dist_labeled)

    local = tamm.LocalTensorDouble([tis])
    local_labeled = local()

    with pytest.raises(ValueError, match="unallocated"):
        local.to_numpy()
    with pytest.raises(ValueError, match="unallocated"):
        tamm.to_numpy(local)
    with pytest.raises(ValueError, match="unallocated"):
        np.asarray(local)

    with pytest.raises(ValueError, match="unallocated"):
        local_labeled.to_numpy()
    with pytest.raises(ValueError, match="unallocated"):
        tamm.to_numpy(local_labeled)
    with pytest.raises(ValueError, match="unallocated"):
        np.asarray(local_labeled)

def make_array(shape, dtype):
    # Deterministic, distinct values so a wrong layout/tiling cannot pass.
    n = int(np.prod(shape, dtype=np.int64))
    base = np.arange(1, n + 1, dtype=np.float64).reshape(shape)
    if np.issubdtype(dtype, np.complexfloating):
        return (base - 0.5j * base + 2.0j).astype(dtype)
    return base.astype(dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize(
    "shape, tilesize",
    [
        ((6,), 2),         # rank 1, evenly divisible
        ((5,), 2),         # rank 1, non-divisible (partial last tile)
        ((4, 4), 2),       # rank 2, divisible
        ((5, 4), 3),       # rank 2, non-divisible
        ((3, 4, 2), 2),    # rank 3
        ((4,), 1),         # tilesize 1
        ((3,), 8),         # tilesize larger than the dimension
        # Larger, rectangular, non-power-of-two sizes with many tiles and a
        # partial last tile in every dimension -- stresses the block-offset /
        # stride arithmetic in copy_numpy_to_tamm_tensor.
        ((17, 31), 4),     # rank 2, many tiles, partial last tile both dims
        ((12, 8, 5), 3),   # rank 3, many tiles, partial last tile
        ((1, 13), 4),      # singleton leading dim + many tiles
    ],
)
# Tests conversions to and from NumPy
def test_from_numpy_to_numpy_roundtrip(ec, dtype, shape, tilesize):
    # from_numpy (eigen_to_tamm equivalent) then to_numpy (tamm_to_eigen
    # equivalent) must reproduce the original array exactly.
    a = make_array(shape, dtype)

    A = tamm.from_numpy(a, ec, tilesize=tilesize)
    try:
        assert dense_shape(A) == shape
        assert_array_matches(A.to_numpy(), a)
    finally:
        deallocate_if_needed(A)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize(
    "layout",
    ["transpose", "fortran", "strided"],
)
# Tests conversions with non-trivial NumPy memory layouts
def test_from_numpy_accepts_non_contiguous_input(ec, dtype, layout):
    # from_numpy relies on pybind11's c_style|forcecast (PyCArray::ensure) to
    # materialize a C-contiguous copy before reading the raw buffer. Feed it
    # non-C-contiguous arrays and confirm the values still round-trip
    if layout == "transpose":
        a = make_array((5, 7), dtype).T            # C-order base -> F-contiguous view
    elif layout == "fortran":
        a = np.asfortranarray(make_array((6, 4), dtype))
    else:  # strided slice (neither C- nor F-contiguous)
        a = make_array((8, 9), dtype)[::2, ::3]

    assert not a.flags.c_contiguous

    A = tamm.from_numpy(a, ec, tilesize=2)
    try:
        assert dense_shape(A) == a.shape
        assert_array_matches(A.to_numpy(), np.ascontiguousarray(a))
    finally:
        deallocate_if_needed(A)


def test_from_numpy_matmul_matches_numpy(ec):
    # End-to-end eigen-equivalent path: build two distributed tensors from
    # NumPy, contract them in TAMM, and check against NumPy's own matmul.
    a = np.arange(12.0).reshape(3, 4)
    b = np.arange(8.0).reshape(4, 2)

    A = tamm.from_numpy(a, ec, tilesize=2)
    B = tamm.from_numpy(b, ec, tilesize=2)
    C = None
    try:
        I, _ = A.tiled_index_spaces()
        _, J = B.tiled_index_spaces()

        C = tamm.TensorDouble([I, J])
        tamm.Scheduler(ec).allocate(C).execute()
        tamm.Scheduler(ec)(C("i", "j"), "=", A("i", "k") * B("k", "j")).execute()
        ec.flush_and_sync()

        assert_array_matches(C.to_numpy(), a @ b)
    finally:
        deallocate_if_needed(A, B, C)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
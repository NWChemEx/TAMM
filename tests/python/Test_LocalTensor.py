#!/usr/bin/env python3

import datetime
import re
import sys

import pytamm as tamm


def atoi_cpp(value):
    s = str(value)
    m = re.match(r"^\s*([+-]?\d+)", s)
    if not m:
        return 0
    return int(m.group(1))


def expects_str(condition, message):
    if not condition:
        raise AssertionError(message)


def check_local_tis_sizes(l_tis, expected_size):
    return (
        int(l_tis.max_num_indices()) == int(expected_size)
        and int(l_tis.tile_size(0)) == int(expected_size)
        and int(l_tis.input_tile_size()) == int(expected_size)
    )


def check_local_tensor_sizes(l_tensor, expected_sizes):
    expects_str(
        int(l_tensor.num_modes()) == len(expected_sizes),
        "Expected sizes should be same as the dimensions of the input LocalTensor.",
    )

    tis_vec = l_tensor.tiled_index_spaces()

    result = True
    for idx, tis in enumerate(tis_vec):
        if not check_local_tis_sizes(tis, expected_sizes[idx]):
            result = False
            break

    return result


def check_local_tensor_values(l_tensor, value):
    expects_str(
        l_tensor.is_allocated(),
        "LocalTensor should be allocated to check the values.",
    )

    tis_sizes = l_tensor.dim_sizes()

    num_elements = 1
    for tis_sz in tis_sizes:
        num_elements *= int(tis_sz)

    local_buf = l_tensor.access_local_buf()

    result = True
    for idx in range(num_elements):
        if local_buf[idx] != value:
            result = False
            break

    return result


def make_local_ec_from(ec):
    pg = ec.pg()
    local_ec = tamm.ExecutionContext(
        pg,
        tamm.DistributionKind.nw,
        tamm.MemoryManagerKind.local,
    )
    return pg, local_ec


def test_local_tensor_constructors(sch, N, tilesize):
    # LocalTensor construction:
    # - TIS list
    # - TIS vec
    # - Labels
    # - Sizes

    tis1 = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(N)),
        tilesize,
    )

    i, j, k, l, m = tis1.labels("all", count=5)

    A = tamm.TensorDouble([i, j, k])
    B = tamm.TensorDouble([k, l])
    C = tamm.TensorDouble([i, j, l])

    sch.allocate(A, B, C).execute()

    expects_str(
        A.is_allocated() and B.is_allocated() and C.is_allocated(),
        "All distributed tensors should be able to allocate!",
    )

    local_pg, local_ec = make_local_ec_from(sch.ec())
    sch_local = tamm.Scheduler(local_ec)

    local_A = tamm.LocalTensorDouble([tis1, tis1, tis1])
    local_B = tamm.LocalTensorDouble(B.tiled_index_spaces())
    local_C = tamm.LocalTensorDouble([i, j, l])
    local_D = tamm.LocalTensorDouble([N, N, N])
    local_E = tamm.LocalTensorDouble([10, 10, 10])

    sch_local.allocate(local_A, local_B, local_C, local_D, local_E).execute()

    expects_str(
        local_A.is_allocated()
        and local_B.is_allocated()
        and local_C.is_allocated()
        and local_D.is_allocated()
        and local_E.is_allocated(),
        "All local tensors should be able to allocate!",
    )

    expects_str(
        check_local_tensor_sizes(local_A, [N, N, N]),
        "Local_A is not correctly created!",
    )
    expects_str(
        check_local_tensor_sizes(local_B, [N, N]),
        "Local_B is not correctly created!",
    )
    expects_str(
        check_local_tensor_sizes(local_C, [N, N, N]),
        "Local_C is not correctly created!",
    )
    expects_str(
        check_local_tensor_sizes(local_D, [N, N, N]),
        "Local_D is not correctly created!",
    )
    expects_str(
        check_local_tensor_sizes(local_E, [10, 10, 10]),
        "Local_E is not correctly created!",
    )

    _ = local_pg


def test_local_tensor_block(ec, N):
    # Block:
    # - Tensor - various sizes
    # - Matrix - various sizes

    local_pg, local_ec = make_local_ec_from(ec)
    sch_local = tamm.Scheduler(local_ec)

    local_A = tamm.LocalTensorDouble([N, N, N])
    local_B = tamm.LocalTensorDouble([N, N])

    sch_local.allocate(local_A, local_B)
    sch_local(local_A(), "=", 42.0)
    sch_local(local_B(), "=", 21.0)
    sch_local.execute()

    local_C = local_A.block([0, 0, 0], [4, 4, 4])
    local_D = local_B.block(0, 0, 4, 4)

    expects_str(
        check_local_tensor_sizes(local_C, [4, 4, 4]),
        "Local_C is not correctly created!",
    )
    expects_str(
        check_local_tensor_sizes(local_D, [4, 4]),
        "Local_D is not correctly created!",
    )
    expects_str(
        check_local_tensor_values(local_C, 42.0),
        "Local_C doesn't have correct values!",
    )
    expects_str(
        check_local_tensor_values(local_D, 21.0),
        "Local_D doesn't have correct values!",
    )

    _ = local_pg


def test_local_tensor_resize(ec, N):
    # Resize:
    # - Smaller
    # - Larger
    # - Same size

    local_pg, local_ec = make_local_ec_from(ec)
    sch_local = tamm.Scheduler(local_ec)

    local_A = tamm.LocalTensorDouble([N, N, N])
    local_B = tamm.LocalTensorDouble([N, N])

    sch_local.allocate(local_A, local_B)
    sch_local(local_A(), "=", 42.0)
    sch_local(local_B(), "=", 21.0)
    sch_local.execute()

    local_A.resize([5, 5, 5])

    expects_str(
        check_local_tensor_sizes(local_A, [5, 5, 5]),
        "Local_A is not correctly created!",
    )
    expects_str(
        check_local_tensor_values(local_A, 42.0),
        "Local_A doesn't have correct values!",
    )

    tensor_ptr = local_A.base_ptr()

    local_A.resize([5, 5, 5])

    tensor_resize_ptr = local_A.base_ptr()

    expects_str(
        tensor_ptr == tensor_resize_ptr,
        "Resize into same size should return the old tensor!",
    )

    local_A.resize([N, N, N])

    expects_str(
        check_local_tensor_sizes(local_A, [N, N, N]),
        "Local_A is not correctly created!",
    )

    expects_str(
        check_local_tensor_values(local_A.block([0, 0, 0], [5, 5, 5]), 42.0),
        "Local_A doesn't have correct values!",
    )

    _ = local_pg


def test_local_tensor_accessor(ec, N):
    # Set/Get:
    # - Single access
    # - Looped access

    local_pg, local_ec = make_local_ec_from(ec)
    sch_local = tamm.Scheduler(local_ec)

    local_A = tamm.LocalTensorDouble([N, N, N])
    local_B = tamm.LocalTensorDouble([N, N])

    sch_local.allocate(local_A, local_B)
    sch_local(local_A(), "=", 42.0)
    sch_local(local_B(), "=", 21.0)
    sch_local.execute()

    expects_str(
        local_A.get(0, 0, 0) == 42.0,
        "The get value doesn't match the expected value.",
    )

    local_A.set([0, 0, 0], 1.0)

    expects_str(
        local_A.get(0, 0, 0) == 1.0,
        "The get value doesn't match the expected value.",
    )

    local_A.set([0, 0, 0], 42.0)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                expects_str(
                    local_A.get(i, j, k) == 42.0,
                    "The get value doesn't match the expected value.",
                )

                local_A.set([i, j, k], local_B.get(i, j))

                expects_str(
                    local_A.get(i, j, k) == 21.0,
                    "The get value doesn't match the expected value.",
                )

    _ = local_pg


def test_local_tensor_copy(ec, N, tilesize):
    local_pg, local_ec = make_local_ec_from(ec)
    sch_local = tamm.Scheduler(local_ec)

    sch_dist = tamm.Scheduler(ec)

    tN = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(N)),
        tilesize,
    )

    dist_A = tamm.TensorDouble([tN, tN, tN])

    sch_dist.allocate(dist_A)
    sch_dist(dist_A(), "=", 42.0)
    sch_dist.execute()

    local_A = tamm.LocalTensorDouble(dist_A.tiled_index_spaces())

    # Copy from distributed tensor
    sch_local.allocate(local_A)
    sch_local(local_A(), "=", 1.0)
    sch_local.execute()

    print("local_A before from_distributed_tensor")
    tamm.print_tensor(local_A)

    local_A.from_distributed_tensor(dist_A)

    print("local_A after from_distributed_tensor")
    tamm.print_tensor(local_A)

    # Copy to distributed tensor
    sch_local(local_A(), "=", 21.0)
    sch_local.execute()

    local_A.to_distributed_tensor(dist_A)

    print("dist_A after to_distributed_tensor")
    if ec.print():
        tamm.print_tensor(dist_A)

    _ = local_pg


def test_local_tensor(sch, N, tilesize):
    tis1 = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(N)),
        tilesize,
    )

    i, j, k, l, m = tis1.labels("all", count=5)

    A = tamm.TensorDouble([i, j, k])
    B = tamm.TensorDouble([k, l])
    C = tamm.TensorDouble([i, j, l])

    sch.allocate(A, B, C)
    sch(A(), "=", 1.0)
    sch(B(), "=", 2.0)
    sch(C(), "=", 3.0)
    sch.execute()

    local_pg, local_ec = make_local_ec_from(sch.ec())
    sch_local = tamm.Scheduler(local_ec)

    new_local1 = tamm.LocalTensorDouble([i, j, k])
    new_local2 = tamm.LocalTensorDouble([tis1, tis1, tis1])
    new_local3 = tamm.LocalTensorDouble([N, N, N])
    new_local4 = tamm.LocalTensorDouble(A.tiled_index_spaces())

    sch_local.allocate(new_local1, new_local2, new_local3, new_local4)
    sch_local(new_local1(), "=", 42.0)
    sch_local(new_local2(), "=", 21.0)
    sch_local(new_local3(), "=", 1.0)
    sch_local(new_local4(), "=", 2.0)
    sch_local.execute()

    new_local3.init(42.0)

    print(f"value at 5,5,5 - {new_local3.get(5, 5, 5)}")

    new_local3.set([5, 5, 5], 1.0)

    val = new_local3.get([5, 5, 5])
    _ = val

    print(f"new value at 5,5,5 - {new_local3.get(5, 5, 5)}")

    print(f"new_local4* before resize - {new_local4.base_ptr()}")
    new_local4.resize([N, N, N])
    print(f"new_local4* after resize - {new_local4.base_ptr()}")

    print("----------------------------------------------------")

    print(f"new_local4* before resize - {new_local4.base_ptr()}")
    new_local4.resize([N + 5, N + 5, N + 5])
    print(f"new_local4* after resize - {new_local4.base_ptr()}")

    new_local5 = new_local3.block([5, 5, 5], [4, 4, 4])

    tamm.print_tensor(new_local1)
    tamm.print_tensor(new_local2)
    tamm.print_tensor(new_local3)
    tamm.print_tensor(new_local4)
    tamm.print_tensor(new_local5)

    _ = local_pg


def print_header(ec, is_size, tile_size):
    if not ec.print():
        return

    print(tamm.tamm_git_info())

    now = datetime.datetime.now()

    print()
    print("date:", now.strftime("%c"))

    print(f"nnodes: {ec.nnodes()}, ", end="")
    print(f"nproc: {ec.nnodes() * ec.ppn()}")

    print(f"dim, tile sizes = {is_size}, {tile_size}")

    ec.print_mem_info()

    print()
    print()


def main(argv=None):
    if argv is None:
        argv = sys.argv

    tamm.initialize(argv, False)

    try:
        if len(argv) < 3:
            raise RuntimeError("Please provide an index space size and tile size")

        is_size = atoi_cpp(argv[1])
        tile_size = atoi_cpp(argv[2])

        if is_size < tile_size:
            tile_size = is_size

        pg = tamm.ProcGroup.create_world_coll()

        ec = tamm.ExecutionContext(
            pg,
            tamm.DistributionKind.nw,
            tamm.MemoryManagerKind.ga,
        )

        ex_hw = ec.exhw()
        _ = ex_hw

        sch = tamm.Scheduler(ec)

        print_header(ec, is_size, tile_size)

        test_local_tensor(sch, is_size, tile_size)
        test_local_tensor_constructors(sch, is_size, tile_size)
        test_local_tensor_copy(ec, is_size, tile_size)
        test_local_tensor_block(ec, is_size)
        test_local_tensor_resize(ec, is_size)
        test_local_tensor_accessor(ec, is_size)

    finally:
        tamm.finalize(True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
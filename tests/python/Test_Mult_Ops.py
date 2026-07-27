#!/usr/bin/env python3
# test_mult_ops_reference.py

import datetime
import re
import sys
import time

import pytamm as tamm


def expects(condition, message="EXPECTS failed"):
    if not condition:
        raise AssertionError(message)


def atoi_cpp(value):
    """
    Approximate std::atoi behavior:
      - skip leading whitespace
      - parse optional sign
      - parse consecutive decimal digits
      - return 0 if no conversion is possible
    """
    s = str(value)
    m = re.match(r"^\s*([+-]?\d+)", s)
    if not m:
        return 0
    return int(m.group(1))


def make_tis(N, tilesize):
    return tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(N)),
        tilesize,
    )


def tensor_double(labels):
    return tamm.TensorDouble(list(labels))


def tensor_complex(labels):
    return tamm.TensorComplexDouble(list(labels))


def norm_check(tensor, ci_check):
    if not ci_check:
        return

    tnorm = tamm.norm(tensor)
    tval = float(tnorm.real) if isinstance(tnorm, complex) else float(tnorm)

    expected = 2.625e8
    mop_pass = abs(tval - expected) <= 1.0e-9

    if not mop_pass:
        ec = tensor.execution_context()
        if ec is not None and ec.print():
            print(f"norm value: {tval}, expected: {expected}")
        expects(mop_pass, "norm_check failed")


def test_2_dim_mult_op(sch, N, tilesize, ex_hw, profile):
    tis1 = make_tis(N, tilesize)
    i, j, k = tis1.labels("all", count=3)

    A = tensor_double([i, k])
    B = tensor_double([k, j])
    C = tensor_double([i, j])

    sch.allocate(A, B, C).execute()

    sch(A(), "=", 21.0)
    sch(B(), "=", 2.0)
    sch(C(), "=", 0.0)
    sch.execute()

    timer_start = time.perf_counter()
    sch(C(j, i), "+=", A(i, k) * B(k, j))
    sch.execute(ex_hw, profile)
    timer_end = time.perf_counter()

    mult_time = timer_end - timer_start

    if sch.ec().print():
        print(
            f"2-D Tensor contraction with {N} indices tiled with "
            f"{tilesize} : {mult_time}"
        )

    sch.deallocate(A, B, C).execute()


def test_3_dim_mult_op(sch, N, tilesize, ex_hw, profile):
    tis1 = make_tis(N, tilesize)
    i, j, k, l, m = tis1.labels("all", count=5)

    A = tensor_double([i, j, l])
    B = tensor_double([l, m, k])
    C = tensor_double([i, j, k])

    sch.allocate(A, B, C)
    sch(A(), "=", 21.0)
    sch(B(), "=", 2.0)
    sch(C(), "=", 0.0)
    sch.execute()

    timer_start = time.perf_counter()
    sch(C(j, i, k), "+=", A(i, j, l) * B(l, m, k))
    sch.execute(ex_hw, profile)
    timer_end = time.perf_counter()

    mult_time = timer_end - timer_start

    if sch.ec().print():
        print(
            f"3-D Tensor contraction with {N} indices tiled with "
            f"{tilesize} : {mult_time}"
        )

    sch.deallocate(A, B, C).execute()


def run_4d_case(
    sch,
    N,
    tilesize,
    ex_hw,
    profile,
    case_name,
    A_factory,
    B_factory,
    C_factory,
    i,
    j,
    k,
    l,
    m,
    o,
):
    A = A_factory([i, j, m, o])
    B = B_factory([m, o, k, l])
    C = C_factory([i, j, k, l])

    sch.allocate(A, B, C)
    sch(A(), "=", 21.0)
    sch(B(), "=", 2.0)
    sch(C(), "=", 0.0)
    sch.execute()

    timer_start = time.perf_counter()
    sch(C(j, i, k, l), "+=", A(i, j, m, o) * B(m, o, k, l))
    sch.execute(ex_hw, profile)
    timer_end = time.perf_counter()

    mult_time = timer_end - timer_start

    if sch.ec().print():
        print(
            f"4-D Tensor contraction ({case_name}) with {N} indices tiled "
            f"with {tilesize} : {mult_time}"
        )

    norm_check(C, N == 50)

    sch.deallocate(A, B, C).execute()


def test_4_dim_mult_op(sch, N, tilesize, ex_hw, profile):
    tis1 = make_tis(N, tilesize)
    i, j, k, l, m, o = tis1.labels("all", count=6)

    # R = R x R
    run_4d_case(
        sch,
        N,
        tilesize,
        ex_hw,
        profile,
        "R=RxR",
        tensor_double,
        tensor_double,
        tensor_double,
        i,
        j,
        k,
        l,
        m,
        o,
    )

    # C = R x R
    run_4d_case(
        sch,
        N,
        tilesize,
        ex_hw,
        profile,
        "C=RxR",
        tensor_double,
        tensor_double,
        tensor_complex,
        i,
        j,
        k,
        l,
        m,
        o,
    )

    # C = R x C
    run_4d_case(
        sch,
        N,
        tilesize,
        ex_hw,
        profile,
        "C=RxC",
        tensor_double,
        tensor_complex,
        tensor_complex,
        i,
        j,
        k,
        l,
        m,
        o,
    )

    # C = C x R
    run_4d_case(
        sch,
        N,
        tilesize,
        ex_hw,
        profile,
        "C=CxR",
        tensor_complex,
        tensor_double,
        tensor_complex,
        i,
        j,
        k,
        l,
        m,
        o,
    )

    # R = C x R
    run_4d_case(
        sch,
        N,
        tilesize,
        ex_hw,
        profile,
        "R=CxR",
        tensor_complex,
        tensor_double,
        tensor_double,
        i,
        j,
        k,
        l,
        m,
        o,
    )

    # R = R x C
    run_4d_case(
        sch,
        N,
        tilesize,
        ex_hw,
        profile,
        "R=RxC",
        tensor_double,
        tensor_complex,
        tensor_double,
        i,
        j,
        k,
        l,
        m,
        o,
    )

    # C = C x C
    run_4d_case(
        sch,
        N,
        tilesize,
        ex_hw,
        profile,
        "C=CxC",
        tensor_complex,
        tensor_complex,
        tensor_complex,
        i,
        j,
        k,
        l,
        m,
        o,
    )


def test_4_dim_mult_op_last_unit(sch, N, tilesize, ex_hw, profile):
    tis1 = make_tis(N, tilesize)

    size = N // 10 if N // 10 > 0 else 1
    tis2 = tamm.TiledIndexSpace(tamm.IndexSpace(tamm.range(size)))

    i, j, k, l = tis1.labels("all", count=4)
    m, o = tis2.labels("all", count=2)

    A = tensor_double([m, o])
    B = tensor_double([i, j, k, m])
    C = tensor_double([i, j, k, m])

    sch.allocate(A, B, C)
    sch(A(), "=", 21.0)
    sch(B(), "=", 2.0)
    sch(C(), "=", 0.0)
    sch.execute()

    timer_start = time.perf_counter()
    sch(C(j, i, k, o), "+=", A(m, o) * B(i, j, k, m))
    sch.execute(ex_hw, profile)
    timer_end = time.perf_counter()

    mult_time = timer_end - timer_start

    if sch.ec().print():
        print(
            f"4-D Tensor contraction with 2-D unit tiled matrix {N} "
            f"indices tiled with {tilesize} last index unit tiled : {mult_time}"
        )

    # Matches C++ reference: no explicit deallocate in this helper.


def test_4_dim_mult_op_first_unit(sch, N, tilesize, ex_hw, profile):
    tis1 = make_tis(N, tilesize)

    size = N // 10 if N // 10 > 0 else 1
    tis2 = tamm.TiledIndexSpace(tamm.IndexSpace(tamm.range(size)))
    tis3 = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(tilesize)),
        tilesize,
    )

    # tis2 is intentionally constructed but unused, matching the C++ reference.
    _ = tis2

    i, j, k, l, m, o = tis1.labels("all", count=6)
    t1, t2 = tis3.labels("all", count=2)

    # Unused labels intentionally preserved.
    _ = (j, k, l, o, t2)

    A = tensor_double([m, t1])
    B = tensor_double([t1, m])
    C = tensor_double([m, i])

    sch.allocate(A, B, C)
    sch(A(), "=", 21.0)
    sch(B(), "=", 2.0)
    sch(C(), "=", 0.0)
    sch.execute()

    timer_start = time.perf_counter()
    sch(C(m, i), "+=", A(m, t1) * B(t1, i))
    sch.execute(ex_hw, profile)
    timer_end = time.perf_counter()

    mult_time = timer_end - timer_start

    if sch.ec().print():
        print(
            f"4-D Tensor contraction with 2-D unit tiled matrix {N} "
            f"indices tiled with {tilesize} first index unit tiled : {mult_time}"
        )

    # Matches C++ reference: no explicit deallocate in this helper.


def print_header(ec, is_size, tile_size):
    if not ec.print():
        return

    print(tamm.tamm_git_info())

    current_time = datetime.datetime.now()

    print()
    print("date:", current_time.strftime("%c"))

    print(f"nnodes: {ec.nnodes()}, ", end="")
    print(f"nproc_per_node: {ec.ppn()}, ", end="")
    print(f"nproc_total: {ec.nnodes() * ec.ppn()}, ", end="")

    if ec.has_gpu():
        if not hasattr(ec, "gpn"):
            raise RuntimeError(
                "ExecutionContext.gpn() must be bound for exact GPU header output"
            )
        print(f"ngpus_per_node: {ec.gpn()}, ", end="")
        print(f"ngpus_total: {ec.nnodes() * ec.gpn()}")

    print()

    ec.print_mem_info()

    print()
    print(f"dim, tile sizes = {is_size}, {tile_size}")
    print()


def write_profile_csv(ec, is_size, tile_size):
    profile_csv = f"multops_profile_{is_size}_{tile_size}.csv"

    try:
        with open(profile_csv, "w") as pds:
            pds.write(str(ec.get_profile_header()))
            pds.write("\n")
            pds.write(str(ec.get_profile_data()))
            pds.write("\n")
    except OSError:
        print(f"Error opening file {profile_csv}", file=sys.stderr)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    tamm.initialize(argv)

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
        sch = tamm.Scheduler(ec)

        print_header(ec, is_size, tile_size)

        profile = True

        # Matches C++ main: these are present in the source but commented out.
        #
        # test_2_dim_mult_op(sch, is_size, tile_size, ex_hw, profile)
        # test_3_dim_mult_op(sch, is_size, tile_size, ex_hw, profile)

        test_4_dim_mult_op(sch, is_size, tile_size, ex_hw, profile)

        # Matches C++ main: these are present in the source but commented out.
        #
        # test_4_dim_mult_op_last_unit(sch, is_size, tile_size, ex_hw, profile)
        # test_4_dim_mult_op_first_unit(sch, is_size, tile_size, ex_hw, profile)

        if profile and ec.print():
            write_profile_csv(ec, is_size, tile_size)

    finally:
        tamm.finalize(True)


if __name__ == "__main__":
    main()
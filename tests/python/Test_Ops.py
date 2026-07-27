#!/usr/bin/env python3

import gc
import itertools
import faulthandler

import pytamm as pt

faulthandler.enable()

TOL = 1.0e-10


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def rank0():
    try:
        return int(pt.ProcGroup.world_rank()) == 0
    except Exception:
        return True


def run_case(name, fn, ec):
    if rank0():
        print(f"[ RUN      ] {name}", flush=True)
    fn(ec)
    if rank0():
        print(f"[       OK ] {name}", flush=True)


def flush_and_sync(ec):
    if hasattr(ec, "flush_and_sync"):
        ec.flush_and_sync()
        return

    try:
        pg = ec.pg()
        if hasattr(pg, "barrier"):
            pg.barrier()
    except Exception:
        pass


def barrier(ec):
    try:
        ec.pg().barrier()
    except Exception:
        flush_and_sync(ec)


def safe_deallocate(*tensors):
    for t in reversed(tensors):
        try:
            if t is not None and hasattr(t, "is_allocated") and t.is_allocated():
                t.deallocate()
        except Exception:
            pass


def allocate_tensors(ec, *tensors):
    doubles = [t for t in tensors if isinstance(t, pt.TensorDouble)]
    complexes = [t for t in tensors if isinstance(t, pt.TensorComplexDouble)]

    if doubles:
        pt.TensorDouble.allocate(ec, *doubles)

    if complexes:
        pt.TensorComplexDouble.allocate(ec, *complexes)


def run_ops(ec, *ops, execute_on=None):
    sch = pt.Scheduler(ec)
    for op in ops:
        sch(*op)

    if execute_on is None:
        sch.execute()
    else:
        sch.execute(execute_on)

    return sch


def run_scheduled(ec, allocate=(), ops=(), deallocate=(), execute_on=None):
    sch = pt.Scheduler(ec)

    if allocate:
        sch.allocate(*allocate)

    for op in ops:
        sch(*op)

    if deallocate:
        sch.deallocate(*deallocate)

    if execute_on is None:
        sch.execute()
    else:
        sch.execute(execute_on)

    return sch


def tensor_cls_is_complex(cls):
    return cls is pt.TensorComplexDouble


def is_complex_obj(obj):
    return isinstance(
        obj,
        (
            pt.TensorComplexDouble,
            pt.LabeledTensorComplexDouble,
        ),
    )


def val_for_cls(cls, x):
    return complex(float(x), 0.0) if tensor_cls_is_complex(cls) else float(x)


def val_for_obj(obj, x):
    return complex(float(x), 0.0) if is_complex_obj(obj) else float(x)


def rank_tensor(cls, tis, rank):
    if rank == 0:
        return cls()
    return cls([tis] * rank)


def as_labeled(obj):
    if isinstance(
        obj,
        (
            pt.LabeledTensorDouble,
            pt.LabeledTensorComplexDouble,
        ),
    ):
        return obj
    return obj()


def check_value(obj, val, label="check_value"):
    lt = as_labeled(obj)
    tensor = lt.tensor()

    ref_real = val.real if isinstance(val, complex) else float(val)

    for itval in pt.LabelLoopNest(lt.labels()):
        blockid = pt.translate_blockid(itval, lt)
        size = int(tensor.block_size(blockid))
        buf = [0] * size
        tensor.get(blockid, buf)

        for x in buf:
            got_real = x.real if isinstance(x, complex) else float(x)
            assert abs(got_real - ref_real) < TOL, (
                label,
                "value mismatch",
                got_real,
                ref_real,
                tuple(blockid),
            )


def all_label_combinations(l1, l2, rank):
    return list(itertools.product([l1, l2], repeat=rank))


def labeled(tensor, labels):
    if len(labels) == 0:
        return tensor()
    return tensor(*labels)


# -----------------------------------------------------------------------------
# Tensor Allocation and Deallocation
# -----------------------------------------------------------------------------

def test_tensor_allocation_and_deallocation(ec):
    tensor = pt.TensorDouble()
    allocate_tensors(ec, tensor)

    # Match C++ scope exit before ec->flush_and_sync().
    del tensor
    gc.collect()

    flush_and_sync(ec)


# -----------------------------------------------------------------------------
# Zero-dimensional ops
# -----------------------------------------------------------------------------

def run_zero_dimensional_gop_block(ec, cleanup=True):
    T1 = pt.TensorDouble()
    T2 = pt.TensorDouble()
    T3 = pt.TensorDouble()

    allocate_tensors(ec, T1, T2, T3)

    # run_ops(
    #     ec,
    #     (T1(), "=", 42.0),
    #     (T2(), "=", 43.0),
    #     (T3(), "=", 44.0),
    # )
    pt.Scheduler(ec)(T1(), "=", 42.0)(T2(), "=", 43.0)(T3(), "=", 44.0).execute()

    # C++:
    # Scheduler{*ec}.gop(T1(), lambda1).execute();
    sch = pt.Scheduler(ec)
    sch.gop_zero_scan(T1())
    sch.execute()
    check_value(T1, 42.0, "zero_dim/gop scan")

    # C++:
    # Scheduler{*ec}.gop(T1(), std::array{T2()}, lambda2).execute();
    sch = pt.Scheduler(ec)
    sch.gop_copy(T1(), T2())
    sch.execute()
    check_value(T1, 43.0, "zero_dim/gop copy")

    # C++:
    # Scheduler{*ec}.gop(T1(), std::array{T2(), T3()}, lambda3).execute();
    sch = pt.Scheduler(ec)
    sch.gop_sum2(T1(), T2(), T3())
    sch.execute()
    check_value(T1, 87.0, "zero_dim/gop sum2")

    if cleanup:
        safe_deallocate(T1, T2, T3)

    return T1, T2, T3


def test_zero_dimensional_ops(ec):
    T = pt.TensorDouble

    # {
    #   Tensor<T> T1{};
    #   Tensor<T>::allocate(ec, T1);
    #   Scheduler{*ec}(T1() = 42).execute();
    #   check_value(T1, 42);
    #   Tensor<T>::deallocate(T1);
    # }
    T1 = T()
    allocate_tensors(ec, T1)
    run_ops(ec, (T1(), "=", 42.0))
    check_value(T1, 42.0, "zero_dim/T1=42")
    T1.deallocate()

    # GOP block
    run_zero_dimensional_gop_block(ec, cleanup=True)

    # Copy scalar
    T1, T2 = T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2),
        ops=[
            (T2(), "=", 42.0),
            (T1(), "=", T2()),
        ],
        deallocate=(T2,),
    )
    check_value(T1, 42.0, "zero_dim/copy")
    T1.deallocate()

    # Add scalar
    T1, T2 = T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2),
        ops=[
            (T1(), "=", 3.0),
            (T2(), "=", 42.0),
            (T1(), "+=", T2()),
        ],
        deallocate=(T2,),
    )
    check_value(T1, 45.0, "zero_dim/add")
    T1.deallocate()

    # Scaled add scalar
    T1, T2 = T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2),
        ops=[
            (T1(), "=", 42.0),
            (T2(), "=", 3.0),
            (T1(), "+=", 2.5 * T2()),
        ],
        deallocate=(T2,),
    )
    check_value(T1, 49.5, "zero_dim/scaled add")
    T1.deallocate()

    # Sub scalar
    T1, T2 = T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2),
        ops=[
            (T1(), "=", 42.0),
            (T2(), "=", 3.0),
            (T1(), "-=", T2()),
        ],
        deallocate=(T2,),
    )
    check_value(T1, 39.0, "zero_dim/sub")
    T1.deallocate()

    # Scaled sub scalar
    T1, T2 = T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2),
        ops=[
            (T1(), "=", 42.0),
            (T2(), "=", 3.0),
            (T1(), "-=", 4.0 * T2()),
        ],
        deallocate=(T2,),
    )
    check_value(T1, 30.0, "zero_dim/scaled sub")
    T1.deallocate()

    # Product scalar
    T1, T2, T3 = T(), T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 0.0),
            (T2(), "=", 3.0),
            (T3(), "=", 5.0),
            (T1(), "+=", T2() * T3()),
        ],
        deallocate=(T2, T3),
    )
    check_value(T1, 15.0, "zero_dim/product")
    T1.deallocate()


def test_zero_dimensional_ops_with_flush_and_sync(ec):
    T = pt.TensorDouble

    # First block, no explicit T1 deallocation before flush.
    T1 = T()
    allocate_tensors(ec, T1)
    run_ops(ec, (T1(), "=", 42.0))
    check_value(T1, 42.0, "zero_dim_flush/T1=42")
    del T1
    gc.collect()

    flush_and_sync(ec)

    # GOP block, then flush.
    T1, T2, T3 = run_zero_dimensional_gop_block(ec, cleanup=False)
    del T1, T2, T3
    gc.collect()

    flush_and_sync(ec)

    # Copy scalar, no explicit T1 deallocation before flush.
    T1, T2 = T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2),
        ops=[
            (T2(), "=", 42.0),
            (T1(), "=", T2()),
        ],
        deallocate=(T2,),
    )
    check_value(T1, 42.0, "zero_dim_flush/copy")
    del T1, T2
    gc.collect()

    flush_and_sync(ec)

    # Add scalar, explicit T1 deallocation.
    T1, T2 = T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2),
        ops=[
            (T1(), "=", 3.0),
            (T2(), "=", 42.0),
            (T1(), "+=", T2()),
        ],
        deallocate=(T2,),
    )
    check_value(T1, 45.0, "zero_dim_flush/add")
    T1.deallocate()
    del T1, T2
    gc.collect()

    # Scaled add scalar, no explicit T1 deallocation before flush.
    T1, T2 = T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2),
        ops=[
            (T1(), "=", 42.0),
            (T2(), "=", 3.0),
            (T1(), "+=", 2.5 * T2()),
        ],
        deallocate=(T2,),
    )
    check_value(T1, 49.5, "zero_dim_flush/scaled add")
    del T1, T2
    gc.collect()

    flush_and_sync(ec)

    # Sub scalar.
    T1, T2 = T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2),
        ops=[
            (T1(), "=", 42.0),
            (T2(), "=", 3.0),
            (T1(), "-=", T2()),
        ],
        deallocate=(T2,),
    )
    check_value(T1, 39.0, "zero_dim_flush/sub")
    del T1, T2
    gc.collect()

    # Scaled sub scalar.
    T1, T2 = T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2),
        ops=[
            (T1(), "=", 42.0),
            (T2(), "=", 3.0),
            (T1(), "-=", 4.0 * T2()),
        ],
        deallocate=(T2,),
    )
    check_value(T1, 30.0, "zero_dim_flush/scaled sub")
    del T1, T2
    gc.collect()

    # Product scalar.
    T1, T2, T3 = T(), T(), T()
    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 0.0),
            (T2(), "=", 3.0),
            (T3(), "=", 5.0),
            (T1(), "+=", T2() * T3()),
        ],
        deallocate=(T2, T3),
    )
    check_value(T1, 15.0, "zero_dim_flush/product")
    del T1, T2, T3
    gc.collect()

    flush_and_sync(ec)


# -----------------------------------------------------------------------------
# Operation helper tests
# -----------------------------------------------------------------------------

def test_setop(ec, T1, LT1, rest_lts=()):
    allocate_tensors(ec, T1)
    try:
        run_ops(
            ec,
            (T1(), "=", val_for_obj(T1, -1.0)),
            (LT1, "=", val_for_obj(T1, 42.0)),
        )
        check_value(LT1, val_for_obj(T1, 42.0), "test_setop/target")

        for idx, lt in enumerate(rest_lts):
            check_value(lt, val_for_obj(T1, -1.0), f"test_setop/rest {idx}")

        return True
    finally:
        safe_deallocate(T1)


def test_mapop(ec, T1, LT1, T2, LT2, rest_lts=()):
    allocate_tensors(ec, T1, T2)
    try:
        sch = pt.Scheduler(ec)
        sch(T1(), "=", val_for_obj(T1, -1.0))
        sch(T2(), "=", val_for_obj(T2, 1.0))
        sch.gop_copy(LT1, LT2)
        sch.execute()

        check_value(LT1, val_for_obj(T1, 1.0), "test_mapop/target")

        for idx, lt in enumerate(rest_lts):
            check_value(lt, val_for_obj(T1, -1.0), f"test_mapop/rest {idx}")

        return True
    finally:
        safe_deallocate(T1, T2)


def test_addop(ec, T1, T2, LT1, LT2, rest_lts=()):
    cls = pt.TensorComplexDouble if is_complex_obj(T1) else pt.TensorDouble
    v = lambda x: val_for_cls(cls, x)

    allocate_tensors(ec, T1, T2)

    try:
        sequences = [
            (
                "copy",
                [
                    (T1(), "=", v(-1.0)),
                    (LT2, "=", v(42.0)),
                    (LT1, "=", LT2),
                ],
                LT1,
                v(42.0),
            ),
            (
                "add",
                [
                    (T1(), "=", v(-1.0)),
                    (LT1, "=", v(4.0)),
                    (LT2, "=", v(42.0)),
                    (LT1, "+=", LT2),
                ],
                LT1,
                v(46.0),
            ),
            (
                "scaled add",
                [
                    (T1(), "=", v(-1.0)),
                    (LT1, "=", v(4.0)),
                    (LT2, "=", v(42.0)),
                    (LT1, "+=", 3.0 * LT2),
                ],
                T1,
                v(130.0),
            ),
            (
                "sub",
                [
                    (T1(), "=", v(-1.0)),
                    (T1(), "=", v(4.0)),
                    (T2(), "=", v(42.0)),
                    (T1(), "-=", T2()),
                ],
                T1,
                v(-38.0),
            ),
            (
                "negative scaled add",
                [
                    (T1(), "=", v(-1.0)),
                    (T1(), "=", v(4.0)),
                    (T2(), "=", v(42.0)),
                    (T1(), "+=", -3.1 * T2()),
                ],
                T1,
                v(-126.2),
            ),
            (
                "scaled sub",
                [
                    (T1(), "=", v(-1.0)),
                    (T1(), "=", v(4.0)),
                    (T2(), "=", v(42.0)),
                    (T1(), "-=", 3.1 * T2()),
                ],
                T1,
                v(-126.2),
            ),
        ]

        for name, ops, target, expected in sequences:
            run_ops(ec, *ops)
            check_value(target, expected, f"test_addop/{name}")

            for idx, lt in enumerate(rest_lts):
                check_value(lt, v(-1.0), f"test_addop/rest {name} {idx}")

        return True
    finally:
        safe_deallocate(T1, T2)


# -----------------------------------------------------------------------------
# setop/mapop parametrized tests
# -----------------------------------------------------------------------------

def setop_cases_for_rank(ec, cls, TIS, l1, l2, rank, target_labels=None, with_rest=False):
    T1 = rank_tensor(cls, TIS, rank)

    if target_labels is None:
        assert test_setop(ec, T1, T1())
        return

    rest = ()
    if with_rest:
        combos = all_label_combinations(l1, l2, rank)
        rest = [labeled(T1, combo) for combo in combos if tuple(combo) != tuple(target_labels)]

    assert test_setop(ec, T1, labeled(T1, target_labels), rest)


def mapop_cases_for_rank(ec, cls, TIS, l1, l2, rank, target_labels=None, with_rest=False):
    T1 = rank_tensor(cls, TIS, rank)
    T2 = rank_tensor(cls, TIS, rank)

    if target_labels is None:
        assert test_mapop(ec, T1, T1(), T2, T2())
        return

    rest = ()
    if with_rest:
        combos = all_label_combinations(l1, l2, rank)
        rest = [labeled(T1, combo) for combo in combos if tuple(combo) != tuple(target_labels)]

    assert test_mapop(ec, T1, labeled(T1, target_labels), T2, labeled(T2, target_labels), rest)


def test_setop_with_type(ec, cls, tilesize):
    IS = pt.IndexSpace(
        pt.range(0, 10),
        {
            "nr1": [pt.range(0, 5)],
            "nr2": [pt.range(5, 10)],
        },
    )
    TIS = pt.TiledIndexSpace(IS, tilesize)

    (l1,) = TIS.labels("nr1", count=1)
    (l2,) = TIS.labels("nr2", count=1)

    T0 = cls()
    assert test_setop(ec, T0, T0())

    cases = [
        (1, None, False),
        (1, (l1,), False),
        (1, (l1,), True),
        (2, None, False),
        (2, (l1, l1), False),
        (2, (l1, l1), True),
        (3, None, False),
        (3, (l1, l2, l2), False),
        (3, (l1, l2, l2), True),
        (4, None, False),
        (4, (l1, l2, l2, l1), False),
        (4, (l1, l2, l2, l1), True),
    ]

    for rank, target, with_rest in cases:
        setop_cases_for_rank(ec, cls, TIS, l1, l2, rank, target, with_rest)


def test_mapop_with_type(ec, cls, tilesize):
    IS = pt.IndexSpace(
        pt.range(0, 10),
        {
            "nr1": [pt.range(0, 5)],
            "nr2": [pt.range(5, 10)],
        },
    )
    TIS = pt.TiledIndexSpace(IS, tilesize)

    (l1,) = TIS.labels("nr1", count=1)
    (l2,) = TIS.labels("nr2", count=1)

    T1 = cls()
    T2 = cls()
    assert test_mapop(ec, T1, T1(), T2, T2())

    cases = [
        (1, None, False),
        (1, (l1,), False),
        (1, (l1,), True),
        (2, None, False),
        (2, (l1, l1), False),
        (2, (l1, l1), True),
        (3, None, False),
        (3, (l1, l2, l2), False),
        (3, (l1, l2, l2), True),
        (4, None, False),
        (4, (l1, l2, l2, l1), False),
        (4, (l1, l2, l2, l1), True),
    ]

    for rank, target, with_rest in cases:
        mapop_cases_for_rank(ec, cls, TIS, l1, l2, rank, target, with_rest)


# -----------------------------------------------------------------------------
# addop parametrized tests
# -----------------------------------------------------------------------------

def basic_addop_case(ec, cls, TIS, rank):
    T1 = rank_tensor(cls, TIS, rank)
    T2 = rank_tensor(cls, TIS, rank)
    assert test_addop(ec, T1, T2, T1(), T2())


def product_tests_for_rank(ec, cls, TIS, rank):
    v = lambda x: val_for_cls(cls, x)

    # T1 rank output, T2 rank RHS, T3 scalar RHS.
    cases = [
        ("plus product rank output", 0.0, 8.0, 4.0, "+=", None, 32.0),
        ("minus product rank output", 0.0, 8.0, 4.0, "-=", None, -32.0),
        ("plus scaled product rank output", 9.0, 8.0, 4.0, "+=", 1.5, 9.0 + 1.5 * 8.0 * 4.0),
        ("minus scaled product rank output", 9.0, 8.0, 4.0, "-=", 1.5, 9.0 - 1.5 * 8.0 * 4.0),
    ]

    for name, init1, init2, init3, opstr, scale, expected in cases:
        T1 = rank_tensor(cls, TIS, rank)
        T2 = rank_tensor(cls, TIS, rank)
        T3 = cls()

        expr = T2() * T3() if scale is None else scale * T3() * T2()

        run_scheduled(
            ec,
            allocate=(T1, T2, T3),
            ops=[
                (T1(), "=", v(init1)),
                (T2(), "=", v(init2)),
                (T3(), "=", v(init3)),
                (T1(), opstr, expr),
            ],
            deallocate=(T2, T3),
        )

        check_value(T1, v(expected), f"product_tests_for_rank/{name}")
        T1.deallocate()

    # Scalar reduction into T3.
    nred = 10 ** rank

    reduction_cases = [
        ("scalar reduction plus", "+=", 4.0 + 1.5 * nred * 9.0 * 8.0),
        ("scalar reduction minus", "-=", 4.0 - 1.5 * nred * 9.0 * 8.0),
    ]

    for name, opstr, expected in reduction_cases:
        T1 = rank_tensor(cls, TIS, rank)
        T2 = rank_tensor(cls, TIS, rank)
        T3 = cls()

        run_scheduled(
            ec,
            allocate=(T1, T2, T3),
            ops=[
                (T1(), "=", v(9.0)),
                (T2(), "=", v(8.0)),
                (T3(), "=", v(4.0)),
                (T3(), opstr, 1.5 * T1() * T2()),
            ],
            deallocate=(T1, T2),
        )

        check_value(T3, v(expected), f"product_tests_for_rank/{name}")
        T3.deallocate()


def test_addop_with_type(ec, cls, tilesize):
    IS = pt.IndexSpace(
        pt.range(0, 10),
        {
            "nr1": [pt.range(0, 5)],
            "nr2": [pt.range(5, 10)],
        },
    )
    TIS = pt.TiledIndexSpace(IS, tilesize)

    for rank in range(5):
        basic_addop_case(ec, cls, TIS, rank)
        product_tests_for_rank(ec, cls, TIS, rank)


# -----------------------------------------------------------------------------
# Two-dimensional ops
# -----------------------------------------------------------------------------

def test_two_dimensional_ops(ec, flush=False):
    IS = pt.IndexSpace(
        pt.range(0, 10),
        {
            "nr1": [pt.range(0, 5)],
            "nr2": [pt.range(5, 10)],
        },
    )
    TIS = pt.TiledIndexSpace(IS, 1)

    (l1,) = TIS.labels("nr1", count=1)
    (l2,) = TIS.labels("nr2", count=1)

    T1 = pt.TensorDouble([TIS, TIS])
    assert test_setop(ec, T1, T1(l1, l1))

    T1 = pt.TensorDouble([TIS, TIS])
    assert test_setop(
        ec,
        T1,
        T1(l1, l1),
        [T1(l1, l2), T1(l2, l1), T1(l2, l2)],
    )

    T1 = pt.TensorDouble([TIS, TIS])
    T2 = pt.TensorDouble([TIS, TIS])
    assert test_addop(ec, T1, T2, T1(), T2())

    # T1 += T2 * T3
    T1 = pt.TensorDouble([TIS, TIS])
    T2 = pt.TensorDouble([TIS, TIS])
    T3 = pt.TensorDouble()
    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 0.0),
            (T2(), "=", 8.0),
            (T3(), "=", 4.0),
            (T1(), "+=", T2() * T3()),
        ],
        deallocate=(T2, T3),
    )
    check_value(T1, 32.0, "two_dim/product")
    if not flush:
        T1.deallocate()
    del T1, T2, T3
    gc.collect()

    # T1 += 1.5 * T3 * T2
    T1 = pt.TensorDouble([TIS, TIS])
    T2 = pt.TensorDouble([TIS, TIS])
    T3 = pt.TensorDouble()
    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 9.0),
            (T2(), "=", 8.0),
            (T3(), "=", 4.0),
            (T1(), "+=", 1.5 * T3() * T2()),
        ],
        deallocate=(T2, T3),
    )
    check_value(T1, 9.0 + 1.5 * 8.0 * 4.0, "two_dim/scaled product")
    if not flush:
        T1.deallocate()
    del T1, T2, T3
    gc.collect()

    # T3 += 1.5 * T1 * T2
    T1 = pt.TensorDouble([TIS, TIS])
    T2 = pt.TensorDouble([TIS, TIS])
    T3 = pt.TensorDouble()
    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 9.0),
            (T2(), "=", 8.0),
            (T3(), "=", 4.0),
            (T3(), "+=", 1.5 * T1() * T2()),
        ],
        deallocate=(T1, T2),
    )
    check_value(T3, 4.0 + 1.5 * 100.0 * 9.0 * 8.0, "two_dim/scalar reduction")
    if not flush:
        T3.deallocate()
    del T1, T2, T3
    gc.collect()

    if flush:
        flush_and_sync(ec)


# -----------------------------------------------------------------------------
# One-dimensional ops
# -----------------------------------------------------------------------------

def test_one_dimensional_ops(ec):
    IS = pt.IndexSpace(
        pt.range(0, 10),
        {
            "nr1": [pt.range(0, 5)],
            "nr2": [pt.range(5, 10)],
        },
    )
    TIS = pt.TiledIndexSpace(IS, 1)

    (l1,) = TIS.labels("nr1", count=1)
    (l2,) = TIS.labels("nr2", count=1)

    T1 = pt.TensorDouble([TIS])
    assert test_setop(ec, T1, T1())

    T1 = pt.TensorDouble([TIS])
    assert test_setop(ec, T1, T1(l1), [T1(l2)])

    T1 = pt.TensorDouble([TIS])
    T2 = pt.TensorDouble([TIS])
    assert test_addop(ec, T1, T2, T1(), T2())

    # T1 += T2 * T3
    T1 = pt.TensorDouble([TIS])
    T2 = pt.TensorDouble([TIS])
    T3 = pt.TensorDouble()
    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 0.0),
            (T2(), "=", 8.0),
            (T3(), "=", 4.0),
            (T1(), "+=", T2() * T3()),
        ],
        deallocate=(T2, T3),
    )
    check_value(T1, 32.0, "one_dim/product")
    T1.deallocate()

    # T1 += 1.5 * T3 * T2
    T1 = pt.TensorDouble([TIS])
    T2 = pt.TensorDouble([TIS])
    T3 = pt.TensorDouble()
    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 9.0),
            (T2(), "=", 8.0),
            (T3(), "=", 4.0),
            (T1(), "+=", 1.5 * T3() * T2()),
        ],
        deallocate=(T2, T3),
    )
    check_value(T1, 9.0 + 1.5 * 8.0 * 4.0, "one_dim/scaled product")
    T1.deallocate()

    # T3 += 1.5 * T1 * T2
    T1 = pt.TensorDouble([TIS])
    T2 = pt.TensorDouble([TIS])
    T3 = pt.TensorDouble()
    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 9.0),
            (T2(), "=", 8.0),
            (T3(), "=", 4.0),
            (T3(), "+=", 1.5 * T1() * T2()),
        ],
        deallocate=(T1, T2),
    )
    check_value(T3, 4.0 + 1.5 * 10.0 * 9.0 * 8.0, "one_dim/scalar reduction")
    T3.deallocate()


# -----------------------------------------------------------------------------
# Three-dimensional mult ops part I
# -----------------------------------------------------------------------------

def test_three_dimensional_mult_ops_part_i(ec):
    IS = pt.IndexSpace(
        pt.range(0, 10),
        {
            "nr1": [pt.range(0, 5)],
            "nr2": [pt.range(5, 10)],
        },
    )
    TIS = pt.TiledIndexSpace(IS, 1)

    i, j, k, l = TIS.labels("all", count=4)

    cases = [
        (
            "mult 3x3x0",
            [TIS, TIS, TIS],
            [TIS, TIS, TIS],
            [],
            lambda T1, T2, T3: (T1(), "+=", 6.9 * T2() * T3()),
            2.0 + 6.9 * 3.0 * 4.0,
        ),
        (
            "mult 3x0x3",
            [TIS, TIS, TIS],
            [TIS, TIS, TIS],
            [],
            lambda T1, T2, T3: (T1(), "+=", 1.7 * T3() * T2()),
            2.0 + 1.7 * 3.0 * 4.0,
        ),
        (
            "mult 3x2x1",
            [TIS, TIS, TIS],
            [TIS, TIS],
            [TIS],
            lambda T1, T2, T3: (T1(i, j, k), "+=", 1.7 * T2(i, j) * T3(k)),
            2.0 + 1.7 * 3.0 * 4.0,
        ),
        (
            "mult 3x3x3",
            [TIS, TIS, TIS],
            [TIS, TIS, TIS],
            [TIS, TIS, TIS],
            lambda T1, T2, T3: (T1(i, j, k), "+=", 1.7 * T2(j, l, i) * T3(l, i, k)),
            2.0 + 1.7 * 3.0 * 4.0 * 10.0,
        ),
    ]

    for name, dims1, dims2, dims3, opbuilder, expected in cases:
        T1 = pt.TensorDouble(dims1)
        T2 = pt.TensorDouble(dims2)
        T3 = pt.TensorDouble(dims3) if dims3 else pt.TensorDouble()

        run_scheduled(
            ec,
            allocate=(T1, T2, T3),
            ops=[
                (T1(), "=", 2.0),
                (T2(), "=", 3.0),
                (T3(), "=", 4.0),
                opbuilder(T1, T2, T3),
            ],
            deallocate=(T2, T3),
        )

        check_value(T1, expected, f"three_dim/{name}")
        T1.deallocate()


# -----------------------------------------------------------------------------
# Four-dimensional mult ops part I
# -----------------------------------------------------------------------------

def test_four_dimensional_mult_ops_part_i(ec):
    IS = pt.IndexSpace(
        pt.range(0, 10),
        {
            "nr1": [pt.range(0, 5)],
            "nr2": [pt.range(5, 10)],
        },
    )
    TIS = pt.TiledIndexSpace(IS, 1)

    i, j, k, l, m, n = TIS.labels("all", count=6)

    cases = [
        (
            "mult 4x4x0",
            [TIS, TIS, TIS, TIS],
            [TIS, TIS, TIS, TIS],
            [],
            lambda T1, T2, T3: (T1(), "+=", 6.9 * T2() * T3()),
            2.0 + 6.9 * 3.0 * 4.0,
        ),
        (
            "mult 4x0x4",
            [TIS, TIS, TIS, TIS],
            [TIS, TIS, TIS, TIS],
            [],
            lambda T1, T2, T3: (T1(), "+=", 1.7 * T3() * T2()),
            2.0 + 1.7 * 3.0 * 4.0,
        ),
        (
            "mult 4x2x2",
            [TIS, TIS, TIS, TIS],
            [TIS, TIS],
            [TIS, TIS],
            lambda T1, T2, T3: (T1(i, j, k, l), "+=", 1.7 * T2(i, j) * T3(k, l)),
            2.0 + 1.7 * 3.0 * 4.0,
        ),
        (
            "mult 4x4x4",
            [TIS, TIS, TIS, TIS],
            [TIS, TIS, TIS, TIS],
            [TIS, TIS, TIS, TIS],
            lambda T1, T2, T3: (
                T1(i, j, k, l),
                "+=",
                1.7 * T2(j, l, k, m) * T3(l, i, k, m),
            ),
            2.0 + 1.7 * 3.0 * 4.0 * 10.0,
        ),
    ]

    for name, dims1, dims2, dims3, opbuilder, expected in cases:
        T1 = pt.TensorDouble(dims1)
        T2 = pt.TensorDouble(dims2)
        T3 = pt.TensorDouble(dims3) if dims3 else pt.TensorDouble()

        run_scheduled(
            ec,
            allocate=(T1, T2, T3),
            ops=[
                (T1(), "=", 2.0),
                (T2(), "=", 3.0),
                (T3(), "=", 4.0),
                opbuilder(T1, T2, T3),
            ],
            deallocate=(T2, T3),
        )

        check_value(T1, expected, f"four_dim/{name}")
        T1.deallocate()


# -----------------------------------------------------------------------------
# Two-dimensional ops part I
# -----------------------------------------------------------------------------

def test_two_dimensional_ops_part_i(ec):
    IS = pt.IndexSpace(
        pt.range(0, 10),
        {
            "nr1": [pt.range(0, 5)],
            "nr2": [pt.range(5, 10)],
        },
    )
    TIS = pt.TiledIndexSpace(IS, 1)

    # setop
    T1 = pt.TensorDouble([TIS, TIS])
    allocate_tensors(ec, T1)
    run_ops(ec, (T1(), "=", 42.0))
    check_value(T1, 42.0, "two_dim_part_i/setop")
    T1.deallocate()

    basic_cases = [
        (
            "copy",
            [
                lambda T1, T2: (T2(), "=", 42.0),
                lambda T1, T2: (T1(), "=", T2()),
            ],
            42.0,
        ),
        (
            "add",
            [
                lambda T1, T2: (T1(), "=", 4.0),
                lambda T1, T2: (T2(), "=", 42.0),
                lambda T1, T2: (T1(), "+=", T2()),
            ],
            46.0,
        ),
        (
            "scaled add",
            [
                lambda T1, T2: (T1(), "=", 4.0),
                lambda T1, T2: (T2(), "=", 42.0),
                lambda T1, T2: (T1(), "+=", 3.0 * T2()),
            ],
            130.0,
        ),
        (
            "sub",
            [
                lambda T1, T2: (T1(), "=", 4.0),
                lambda T1, T2: (T2(), "=", 42.0),
                lambda T1, T2: (T1(), "-=", T2()),
            ],
            -38.0,
        ),
        (
            "negative scaled add",
            [
                lambda T1, T2: (T1(), "=", 4.0),
                lambda T1, T2: (T2(), "=", 42.0),
                lambda T1, T2: (T1(), "+=", -3.1 * T2()),
            ],
            -126.2,
        ),
    ]

    for name, builders, expected in basic_cases:
        T1 = pt.TensorDouble([TIS, TIS])
        T2 = pt.TensorDouble([TIS, TIS])

        run_scheduled(
            ec,
            allocate=(T1, T2),
            ops=[builder(T1, T2) for builder in builders],
            deallocate=(T2,),
        )

        check_value(T1, expected, f"two_dim_part_i/{name}")
        T1.deallocate()

    # multop: 2,2,0 plus/minus
    for name, opstr, expected in [
        ("mult plus", "+=", 4.0 - 3.1 * 42.0 * 5.0),
        ("mult minus", "-=", 4.0 + 3.1 * 42.0 * 5.0),
    ]:
        T1 = pt.TensorDouble([TIS, TIS])
        T2 = pt.TensorDouble([TIS, TIS])
        T3 = pt.TensorDouble()

        run_scheduled(
            ec,
            allocate=(T1, T2, T3),
            ops=[
                (T1(), "=", 4.0),
                (T2(), "=", 42.0),
                (T3(), "=", 5.0),
                (T1(), opstr, -3.1 * T2() * T3()),
            ],
            deallocate=(T2, T3),
        )

        check_value(T1, expected, f"two_dim_part_i/{name}")
        T1.deallocate()

    # scalar reduction
    T1 = pt.TensorDouble()
    T2 = pt.TensorDouble([TIS, TIS])
    T3 = pt.TensorDouble([TIS, TIS])

    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 4.0),
            (T2(), "=", 42.0),
            (T3(), "=", 5.0),
            (T1(), "+=", -3.1 * T2() * T3()),
        ],
        deallocate=(T2, T3),
    )

    check_value(T1, 4.0 - 3.1 * 10.0 * 10.0 * 42.0 * 5.0, "two_dim_part_i/scalar reduction")
    T1.deallocate()

    i, j, k = TIS.labels("all", count=3)

    # multop 2,1,1
    T1 = pt.TensorDouble([TIS, TIS])
    T2 = pt.TensorDouble([TIS])
    T3 = pt.TensorDouble([TIS])

    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 4.0),
            (T2(), "=", 42.0),
            (T3(), "=", 5.0),
            (T1(i, j), "+=", -3.1 * T2(i) * T3(j)),
        ],
        deallocate=(T2, T3),
    )

    check_value(T1, 4.0 - 3.1 * 42.0 * 5.0, "two_dim_part_i/2x1x1")
    T1.deallocate()

    # multop 2,2,1
    expr_cases = [
        ("T2(i,j)*T3(i)", lambda T2, T3: T2(i, j) * T3(i)),
        ("T2(j,i)*T3(i)", lambda T2, T3: T2(j, i) * T3(i)),
        ("T2(i,j)*T3(j)", lambda T2, T3: T2(i, j) * T3(j)),
        ("T3(j)*T2(j,i)", lambda T2, T3: T3(j) * T2(j, i)),
    ]

    for name, expr_builder in expr_cases:
        T1 = pt.TensorDouble([TIS, TIS])
        T2 = pt.TensorDouble([TIS, TIS])
        T3 = pt.TensorDouble([TIS])

        run_scheduled(
            ec,
            allocate=(T1, T2, T3),
            ops=[
                (T1(), "=", 4.0),
                (T2(), "=", 42.0),
                (T3(), "=", 5.0),
                (T1(i, j), "+=", -3.1 * expr_builder(T2, T3)),
            ],
            deallocate=(T2, T3),
        )

        check_value(T1, 4.0 - 3.1 * 42.0 * 5.0, f"two_dim_part_i/{name}")
        T1.deallocate()

    # multop 2,2,2 no reduction
    T1 = pt.TensorDouble([TIS, TIS])
    T2 = pt.TensorDouble([TIS, TIS])
    T3 = pt.TensorDouble([TIS, TIS])

    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 4.0),
            (T2(), "=", 42.0),
            (T3(), "=", 5.0),
            (T1(i, j), "+=", -3.1 * T2(i, j) * T3(i, j)),
        ],
        deallocate=(T2, T3),
    )

    check_value(T1, 4.0 - 3.1 * 42.0 * 5.0, "two_dim_part_i/2x2x2 no reduction")
    T1.deallocate()

    # multop 2,2,2 with reduction
    T1 = pt.TensorDouble([TIS, TIS])
    T2 = pt.TensorDouble([TIS, TIS])
    T3 = pt.TensorDouble([TIS, TIS])

    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 4.0),
            (T2(), "=", 42.0),
            (T3(), "=", 5.0),
            (T1(i, j), "+=", -3.1 * T2(i, k) * T3(j, k)),
        ],
        deallocate=(T2, T3),
    )

    check_value(T1, 4.0 - 3.1 * 42.0 * 5.0 * 10.0, "two_dim_part_i/2x2x2 reduction")
    T1.deallocate()


# -----------------------------------------------------------------------------
# MultOp with RHS reduction
# -----------------------------------------------------------------------------

def test_multop_with_rhs_reduction(ec):
    IS = pt.IndexSpace(
        pt.range(0, 4),
        {
            "nr1": [pt.range(0, 2)],
            "nr2": [pt.range(2, 4)],
        },
    )
    TIS = pt.TiledIndexSpace(IS, 1)

    i, j, k = TIS.labels("all", count=3)

    # T1 += -3.1 * T2(i) * T3()
    T1 = pt.TensorDouble()
    T2 = pt.TensorDouble([TIS])
    T3 = pt.TensorDouble()

    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 4.0),
            (T2(), "=", 3.0),
            (T3(), "=", 5.0),
            (T1(), "+=", -3.1 * T2(i) * T3()),
        ],
        deallocate=(T2, T3),
    )

    check_value(T1, 4.0 - 3.1 * 4.0 * 3.0 * 5.0, "rhs_reduction/1")
    T1.deallocate()

    # T1 += -3.1 * T2() * T3(j)
    T1 = pt.TensorDouble()
    T2 = pt.TensorDouble()
    T3 = pt.TensorDouble([TIS])

    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 4.0),
            (T2(), "=", 3.0),
            (T3(), "=", 5.0),
            (T1(), "+=", -3.1 * T2() * T3(j)),
        ],
        deallocate=(T2, T3),
    )

    check_value(T1, 4.0 - 3.1 * 4.0 * 3.0 * 5.0, "rhs_reduction/2")
    T1.deallocate()

    # T1 += -3.2 * T2(i) * T3(j)
    T1 = pt.TensorDouble()
    T2 = pt.TensorDouble([TIS])
    T3 = pt.TensorDouble([TIS])

    run_scheduled(
        ec,
        allocate=(T1, T2, T3),
        ops=[
            (T1(), "=", 4.0),
            (T2(), "=", 3.0),
            (T3(), "=", 5.0),
            (T1(), "+=", -3.2 * T2(i) * T3(j)),
        ],
        deallocate=(T2, T3),
    )

    check_value(T1, 4.0 - 3.2 * 4.0 * 4.0 * 3.0 * 5.0, "rhs_reduction/3")
    T1.deallocate()

    # Larger RHS-reduction comparison
    cvec = pt.IndexSpace(pt.range(0, 4))
    CV = pt.TiledIndexSpace(cvec, 1)

    MO_IS = pt.IndexSpace(
        pt.range(0, 14),
        {
            "occ": [pt.range(0, 10)],
            "virt": [pt.range(10, 14)],
        },
    )
    MO = pt.TiledIndexSpace(MO_IS, [5, 5, 2, 2])

    O = MO("occ")
    V = MO("virt")
    N = MO("all")

    CV3D = pt.TensorDouble([N, N, CV])
    CV2D = pt.TensorDouble([N, N])
    res = pt.TensorDouble([O, O])
    res1 = pt.TensorDouble([O, O])
    tmp1 = pt.TensorDouble([O, O])
    tmp2 = pt.TensorDouble([O, O])
    t1 = pt.TensorDouble([V, O])
    sc1 = pt.TensorDouble()
    sc2 = pt.TensorDouble()

    allocate_tensors(ec, sc1, sc2, res, res1, tmp1, tmp2, t1, CV3D, CV2D)

    sch = pt.Scheduler(ec)
    sch(CV3D(), "=", 42.0)
    sch(CV2D(), "=", 42.0)
    sch(t1(), "=", 2.0)
    sch(res(), "=", 0.0)
    sch(res1(), "=", 0.0)
    sch(sc1(), "=", 0.0)
    sch(sc2(), "=", 0.0)
    sch(tmp1(), "=", 0.0)
    sch(tmp2(), "=", 0.0)
    sch.execute()

    # The C++ code reads CV2D/CV3D into temporary vectors and mutates the vectors
    # without putting them back. That has no effect on the tensor state, so this
    # Python translation intentionally preserves the same effective behavior.

    barrier(ec)

    (cind,) = CV.labels("all", count=1)
    p1, p2, p3 = MO.labels("virt", count=3)
    h1, h2, h3 = MO.labels("occ", count=3)

    sch(res(h2, h1), "+=", 1.0 * t1(p1, h1) * CV3D(h2, p1, cind))
    sch(sc1(), "+=", 1.0 * t1(p1, h1) * CV3D(h1, p1, cind))
    sch(tmp1(h2, h1), "+=", -1.0 * res(h3, h1) * res(h2, h3))

    for _ in range(4):
        sch(res1(h2, h1), "+=", 1.0 * t1(p1, h1) * CV2D(h2, p1))
        sch(sc2(), "+=", 1.0 * t1(p1, h1) * CV2D(h1, p1))

    sch(tmp2(h2, h1), "+=", -1.0 * res1(h3, h1) * res1(h2, h3))
    sch.execute()

    for blockid in tmp2.loop_nest():
        size = int(tmp2.block_size(blockid))
        buf2 = [0.0] * size
        buf1 = [0.0] * size

        tmp2.get(blockid, buf2)
        tmp1.get(blockid, buf1)

        for x, y in zip(buf2, buf1):
            assert x == y, ("rhs_reduction/final exact compare", tuple(blockid), x, y)

    barrier(ec)

    safe_deallocate(sc1, sc2, res, res1, tmp1, tmp2, t1, CV3D, CV2D)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    pt.initialize(["Test_Ops.py"], False)

    tests = [
        ("Tensor Allocation and Deallocation", test_tensor_allocation_and_deallocation),
        ("Zero-dimensional ops", test_zero_dimensional_ops),
        ("Zero-dimensional ops with flush and sync deallocation", test_zero_dimensional_ops_with_flush_and_sync),
        ("setop with double", lambda ec: test_setop_with_type(ec, pt.TensorDouble, 1)),
        ("mapop with double tile=1", lambda ec: test_mapop_with_type(ec, pt.TensorDouble, 1)),
        ("mapop with double tile=3", lambda ec: test_mapop_with_type(ec, pt.TensorDouble, 3)),
        ("addop with double tile=1", lambda ec: test_addop_with_type(ec, pt.TensorDouble, 1)),
        ("addop with double tile=3", lambda ec: test_addop_with_type(ec, pt.TensorDouble, 3)),
        ("addop with double complex tile=1", lambda ec: test_addop_with_type(ec, pt.TensorComplexDouble, 1)),
        ("addop with double complex tile=3", lambda ec: test_addop_with_type(ec, pt.TensorComplexDouble, 3)),
        ("Two-dimensional ops", lambda ec: test_two_dimensional_ops(ec, flush=False)),
        ("Two-dimensional ops with flush and sync", lambda ec: test_two_dimensional_ops(ec, flush=True)),
        ("One-dimensional ops", test_one_dimensional_ops),
        ("Three-dimensional mult ops part I", test_three_dimensional_mult_ops_part_i),
        ("Four-dimensional mult ops part I", test_four_dimensional_mult_ops_part_i),
        ("Two-dimensional ops part I", test_two_dimensional_ops_part_i),
        ("MultOp with RHS reduction", test_multop_with_rhs_reduction),
    ]

    try:
        for name, fn in tests:
            pg = pt.ProcGroup.create_world_coll()
            ec = pt.ExecutionContext(pg, pt.DistributionKind.nw, pt.MemoryManagerKind.ga)

            try:
                run_case(name, fn, ec)
            finally:
                del ec
                del pg
                gc.collect()

    finally:
        pt.finalize(True)


if __name__ == "__main__":
    main()
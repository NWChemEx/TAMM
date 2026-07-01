#!/usr/bin/env python3

import gc
import sys
import faulthandler

import pytamm as pt

faulthandler.enable()

TOL = 1.0e-10


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def rank0():
    try:
        return int(pt.ProcGroup.world_rank()) == 0
    except Exception:
        return True


def rprint(*args, **kwargs):
    if rank0():
        print(*args, **kwargs)


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


def run_ops(ec, *ops):
    sch = pt.Scheduler(ec)
    for op in ops:
        sch(*op)
    sch.execute()
    return sch


def run_scheduled(ec, allocate=(), ops=(), deallocate=()):
    sch = pt.Scheduler(ec)

    if allocate:
        sch.allocate(*allocate)

    for op in ops:
        sch(*op)

    if deallocate:
        sch.deallocate(*deallocate)

    sch.execute()
    return sch


def as_labeled(obj):
    if isinstance(obj, (pt.LabeledTensorDouble, pt.LabeledTensorComplexDouble)):
        return obj
    return obj()


def check_value(obj, val):
    lt = as_labeled(obj)
    tensor = lt.tensor()
    ref = float(val.real) if isinstance(val, complex) else float(val)

    for itval in pt.LabelLoopNest(lt.labels()):
        blockid = pt.translate_blockid(itval, lt)
        size = int(tensor.block_size(blockid))
        buf = [0.0] * size
        tensor.get(blockid, buf)

        for x in buf:
            got = float(x.real) if isinstance(x, complex) else float(x)
            assert abs(got - ref) < TOL, (
                "value mismatch",
                got,
                ref,
                tuple(blockid),
            )


def tensor_from_labels(labels):
    return pt.TensorDouble(list(labels))


def labeled_tensor(tensor, labels):
    return tensor(*list(labels))


def construction_check(labels):
    _ = tensor_from_labels(labels)


def allocation_check(ec, labels):
    t = tensor_from_labels(labels)
    try:
        allocate_tensors(ec, t)
    finally:
        safe_deallocate(t)


def basic_setop_check(ec, labels):
    t = tensor_from_labels(labels)
    try:
        allocate_tensors(ec, t)
        run_ops(ec, (t(), "=", 42.0))
        check_value(t, 42.0)
    finally:
        safe_deallocate(t)


# -----------------------------------------------------------------------------
# C++ helper equivalents: test_setop / test_addop
# -----------------------------------------------------------------------------

def test_setop(ec, T1, LT1, rest_lts=()):
    success = True

    try:
        allocate_tensors(ec, T1)

        try:
            run_ops(
                ec,
                (T1(), "=", -1.0),
                (LT1, "=", 42.0),
            )

            check_value(LT1, 42.0)

            for lt in rest_lts:
                check_value(lt, -1.0)

        except Exception as exc:
            print(f"Caught exception: {exc}", file=sys.stderr, flush=True)
            success = False

        safe_deallocate(T1)

    except Exception as exc:
        print(f"Caught exception: {exc}", file=sys.stderr, flush=True)
        success = False
        safe_deallocate(T1)

    return success


def test_addop(ec, T1, T2, LT1, LT2, rest_lts=()):
    success = True

    allocate_tensors(ec, T1, T2)

    try:
        try:
            run_ops(
                ec,
                (T1(), "=", -1.0),
                (LT2, "=", 42.0),
                (LT1, "=", LT2),
            )

            check_value(LT1, 42.0)

            for lt in rest_lts:
                check_value(lt, -1.0)

        except Exception as exc:
            print(f"AddOp. Test 0. Exception: {exc}", file=sys.stderr, flush=True)
            success = False

        try:
            success = True

            run_ops(
                ec,
                (T1(), "=", -1.0),
                (LT1, "=", 4.0),
                (LT2, "=", 42.0),
                (LT1, "+=", LT2),
            )

            check_value(LT1, 46.0)

            for lt in rest_lts:
                check_value(lt, -1.0)

        except Exception as exc:
            print(f"AddOp. Test 1. Exception: {exc}", file=sys.stderr, flush=True)
            success = False

        try:
            success = True

            run_ops(
                ec,
                (T1(), "=", -1.0),
                (LT1, "=", 4.0),
                (LT2, "=", 42.0),
                (LT1, "+=", 3.0 * LT2),
            )

            check_value(T1, 130.0)

            for lt in rest_lts:
                check_value(lt, -1.0)

        except Exception as exc:
            print(f"AddOp. Test 2. Exception: {exc}", file=sys.stderr, flush=True)
            success = False

        try:
            success = True

            run_ops(
                ec,
                (T1(), "=", -1.0),
                (T1(), "=", 4.0),
                (T2(), "=", 42.0),
                (T1(), "-=", T2()),
            )

            check_value(T1, -38.0)

            for lt in rest_lts:
                check_value(lt, -1.0)

        except Exception as exc:
            print(f"AddOp. Test 3. Exception: {exc}", file=sys.stderr, flush=True)
            success = False

        try:
            success = True

            run_ops(
                ec,
                (T1(), "=", -1.0),
                (T1(), "=", 4.0),
                (T2(), "=", 42.0),
                (T1(), "+=", -3.1 * T2()),
            )

            check_value(T1, -126.2)

            for lt in rest_lts:
                check_value(lt, -1.0)

        except Exception as exc:
            print(f"AddOp. Test 4. Exception: {exc}", file=sys.stderr, flush=True)
            success = False

    finally:
        safe_deallocate(T1, T2)

    return success


# -----------------------------------------------------------------------------
# Dependent-space construction
# -----------------------------------------------------------------------------

def make_dependent_index_space(base_is, base_tis, tilesize):
    tile_count = 10 // int(tilesize)
    if 10 % int(tilesize) > 0:
        tile_count += 1

    dep_relation = {}
    for idx in range(tile_count):
        dep_relation[(idx,)] = base_is

    # Exact C++ equivalent:
    #   IndexSpace DIS{{T_IS}, dep_relation};
    return pt.IndexSpace([base_tis], dep_relation)


# -----------------------------------------------------------------------------
# Dependent-space multop checks
# -----------------------------------------------------------------------------

def multop_output_check(ec, labels_factory, expected_rank_factor):
    # MultOp N-dim += N-dim * 0-dim
    labels = labels_factory()
    T1 = tensor_from_labels(labels)
    T2 = tensor_from_labels(labels)
    T3 = pt.TensorDouble()

    try:
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

        check_value(T1, 32.0)

    finally:
        safe_deallocate(T1, T2, T3)

    # MultOp N-dim += alpha * 0-dim * N-dim
    labels = labels_factory()
    T1 = tensor_from_labels(labels)
    T2 = tensor_from_labels(labels)
    T3 = pt.TensorDouble()

    try:
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

        check_value(T1, 9.0 + 1.5 * 8.0 * 4.0)

    finally:
        safe_deallocate(T1, T2, T3)

    # MultOp 0-dim += alpha * N-dim * N-dim
    labels = labels_factory()
    T1 = tensor_from_labels(labels)
    T2 = tensor_from_labels(labels)
    T3 = pt.TensorDouble()

    try:
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

        check_value(T3, 4.0 + 1.5 * expected_rank_factor * 9.0 * 8.0)

    finally:
        safe_deallocate(T1, T2, T3)


# -----------------------------------------------------------------------------
# Per-rank dependent-space suite
# -----------------------------------------------------------------------------

def dependent_rank_suite(
    ec,
    labels_factory,
    expected_rank_factor,
    no_label_setop_labels_factory=None,
    print_message_before_scalar_reduction=False,
    print_finished_message=False,
):
    if no_label_setop_labels_factory is None:
        no_label_setop_labels_factory = labels_factory

    # Tensor Construction
    labels = labels_factory()
    construction_check(labels)

    # Tensor Allocation / Deallocate
    labels = labels_factory()
    allocation_check(ec, labels)

    # Basic SetOp
    labels = labels_factory()
    basic_setop_check(ec, labels)

    # SetOp test with no labels
    labels = no_label_setop_labels_factory()
    T1 = tensor_from_labels(labels)
    assert test_setop(ec, T1, T1())

    # SetOp test with labels provided
    labels = labels_factory()
    T1 = tensor_from_labels(labels)
    assert test_setop(ec, T1, labeled_tensor(T1, labels))

    # AddOp test with no labels
    labels = labels_factory()
    T1 = tensor_from_labels(labels)
    T2 = tensor_from_labels(labels)
    assert test_addop(ec, T1, T2, T1(), T2())

    # AddOp test with labels
    labels = labels_factory()
    T1 = tensor_from_labels(labels)
    T2 = tensor_from_labels(labels)
    assert test_addop(ec, T1, T2, labeled_tensor(T1, labels), labeled_tensor(T2, labels))

    # MultOp N-dim += N-dim * 0-dim
    labels = labels_factory()
    T1 = tensor_from_labels(labels)
    T2 = tensor_from_labels(labels)
    T3 = pt.TensorDouble()

    try:
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

        check_value(T1, 32.0)

    finally:
        safe_deallocate(T1, T2, T3)

    # MultOp N-dim += alpha * 0-dim * N-dim
    labels = labels_factory()
    T1 = tensor_from_labels(labels)
    T2 = tensor_from_labels(labels)
    T3 = pt.TensorDouble()

    try:
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

        check_value(T1, 9.0 + 1.5 * 8.0 * 4.0)

    finally:
        safe_deallocate(T1, T2, T3)

    if print_message_before_scalar_reduction:
        print("/* message */", file=sys.stderr, flush=True)

    # MultOp 0-dim += alpha * N-dim * N-dim
    labels = labels_factory()
    T1 = tensor_from_labels(labels)
    T2 = tensor_from_labels(labels)
    T3 = pt.TensorDouble()

    try:
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

        check_value(T3, 4.0 + 1.5 * expected_rank_factor * 9.0 * 8.0)

    finally:
        safe_deallocate(T1, T2, T3)

    if print_finished_message:
        print("Finished default dependent space", file=sys.stderr, flush=True)


# -----------------------------------------------------------------------------
# Main dependent-space test
# -----------------------------------------------------------------------------

def test_dependent_space_with_double(tilesize):
    pg = pt.ProcGroup.create_world_coll()
    ec = pt.ExecutionContext(pg, pt.DistributionKind.nw, pt.MemoryManagerKind.ga)

    try:
        IS = pt.IndexSpace(pt.range(0, 10))
        T_IS = pt.TiledIndexSpace(IS, tilesize)

        DIS = make_dependent_index_space(IS, T_IS, tilesize)
        T_DIS = pt.TiledIndexSpace(DIS, tilesize)

        a, b = T_DIS.labels("all", count=2)
        i, j = T_IS.labels("all", count=2)

        def labels_2d():
            # Tensor<T> T1{a(i), i};
            return [a(i), i]

        def labels_3d():
            # Tensor<T> T1{a(i), i, j};
            return [a(i), i, j]

        def labels_4d():
            # Tensor<T> T1{a(i), i, b(j), j};
            return [a(i), i, b(j), j]

        dependent_rank_suite(
            ec,
            labels_factory=labels_2d,
            expected_rank_factor=100,
        )

        dependent_rank_suite(
            ec,
            labels_factory=labels_3d,
            expected_rank_factor=1000,
            print_message_before_scalar_reduction=True,
            print_finished_message=True,
        )

        dependent_rank_suite(
            ec,
            labels_factory=labels_4d,
            expected_rank_factor=10000,
            no_label_setop_labels_factory=labels_3d,
            print_finished_message=True,
        )

    finally:
        del ec
        del pg
        gc.collect()


def main():
    pt.initialize(["Test_DependentSpace.py"], False)

    try:
        rprint("[ RUN      ] Tensor ops for double / tilesize=1", flush=True)
        test_dependent_space_with_double(1)
        rprint("[       OK ] Tensor ops for double / tilesize=1", flush=True)

        rprint("[ RUN      ] Tensor ops for double / tilesize=3", flush=True)
        test_dependent_space_with_double(3)
        rprint("[       OK ] Tensor ops for double / tilesize=3", flush=True)

    finally:
        pt.finalize(True)


if __name__ == "__main__":
    main()
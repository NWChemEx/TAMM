# pytamm (Python) half of the binding-equivalence check.
#
# Runs a fixed set of tensor operations through the pytamm bindings and writes
# the results to disk under the `py_tamm_equiv` prefix.
#
# Part of a 3-file check proving the Python bindings produce numerically
# identical results to core C++ TAMM:
#   1. Test_Binding_Equivalence.cpp    -> writes `cpp_tamm_equiv` (same ops, C++)
#   2. Test_Binding_Equivalence.py     -> writes `py_tamm_equiv`  (this file)
#   3. Compare_Binding_Equivalence.py  -> reads both, asserts they match
# Run 1 and 2 (order-independent), then 3.

import argparse
import os
import sys

import pytamm as pt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="py_tamm_equiv")
    args, _ = p.parse_known_args()
    return args


def nz_a(b):
    x, y = int(b[0]), int(b[1])
    return ((x + 2 * y) % 3) != 1


def nz_b(b):
    x, y = int(b[0]), int(b[1])
    return ((2 * x + y) % 3) != 2


def nz_d(b):
    x, y = int(b[0]), int(b[1])
    return x == y or ((x + y) % 2 == 0)


def nz_c(b):
    x, y = int(b[0]), int(b[1])
    return x <= y or ((x + y) % 2 == 0)


def dval(which, bid, p):
    x, y, q = int(bid[0]), int(bid[1]), int(p)

    if which == 0:
        n = 16 * x + 4 * y + q + 1
    elif which == 1:
        n = 8 * x - 6 * y + q + 3
    else:
        n = 5 * x + 7 * y + 2 * q + 1

    return n / 8.0


def zval(which, bid, p):
    x, y, q = int(bid[0]), int(bid[1]), int(p)

    if which == 0:
        re = 6 * x + 2 * y + q + 1
        im = 3 * x - 5 * y + q + 2
    else:
        re = -4 * x + 7 * y + q + 1
        im = 5 * x + y - q - 1

    return complex(re / 8.0, im / 8.0)


def fill_double_tensor(tensor, which):
    def fill(blockid, buf):
        for p in range(buf.size):
            buf[p] = dval(which, blockid, p)

    pt.fill_sparse_tensor_double(tensor, fill)


def fill_complex_tensor(tensor, which):
    def fill(blockid, buf):
        for p in range(buf.size):
            buf[p] = zval(which, blockid, p)

    pt.fill_sparse_tensor_complex_double(tensor, fill)


def make_spaces():
    ispace = pt.IndexSpace(pt.range(6))
    return pt.TiledIndexSpace(ispace, 2)


def make_tensors(x):
    info_a = pt.TensorInfo([x, x], nz_a)
    info_b = pt.TensorInfo([x, x], nz_b)
    info_d = pt.TensorInfo([x, x], nz_d)
    info_c = pt.TensorInfo([x, x], nz_c)

    A = pt.TensorDouble([x, x], info_a)
    B = pt.TensorDouble([x, x], info_b)
    D = pt.TensorDouble([x, x], info_d)
    C = pt.TensorDouble([x, x], info_c)
    R = pt.TensorDouble([x, x], info_c)

    ZA = pt.TensorComplexDouble([x, x], info_a)
    ZB = pt.TensorComplexDouble([x, x], info_b)
    ZC = pt.TensorComplexDouble([x, x], info_c)

    return A, B, D, C, R, ZA, ZB, ZC


def run_scheduler_double(ec, i, j, k, A, B, D, C):
    sch = pt.Scheduler(ec)
    sch(C(i, j), "=", 0.0)
    sch(C(i, j), "+=", 1.5, A(i, k), B(k, j))
    sch(C(i, j), "-=", 0.25, D(i, j))
    sch.execute(pt.ExecutionHW.CPU, False)


def run_scheduler_complex(ec, i, j, k, ZA, ZB, ZC):
    alpha = complex(0.5, -0.25)
    beta = complex(-0.125, 0.25)

    sch = pt.Scheduler(ec)
    sch(ZC(i, j), "=", 0.0 + 0.0j)
    sch(ZC(i, j), "+=", alpha, ZA(i, k), ZB(k, j))
    sch(ZC(i, j), "+=", beta, ZA(i, j))
    sch.execute(pt.ExecutionHW.CPU, False)


def run_opmin_double(ec, i, j, k, A, B, D, R):
    pt.Scheduler(ec)(R(i, j), "=", 0.0).execute(pt.ExecutionHW.CPU, False)

    rhs = (
        1.25 * pt.as_op(A(i, k)) * pt.as_op(B(k, j))
        + (-0.5) * pt.as_op(D(i, j))
    )

    R(i, j).update(rhs)

    st = pt.SymbolTable()
    pt.register_symbols(st, A=A, B=B, D=D, R=R, i=i, j=j, k=k)

    sch = pt.Scheduler(ec)
    ex = pt.OpExecutor(sch, st)
    ex.opmin_execute(R)


def run(prefix):
    pg = pt.ProcGroup.create_world_coll()
    ec = pt.ExecutionContext(pg, pt.DistributionKind.nw, pt.MemoryManagerKind.ga)

    if ec.pg_rank() == 0:
        dirname = os.path.dirname(os.path.abspath(prefix))
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    pg.barrier()

    x = make_spaces()
    i, j, k = x.labels("all", None, 3)

    A, B, D, C, R, ZA, ZB, ZC = make_tensors(x)

    pt.TensorDouble.allocate(ec, A, B, D, C, R)
    pt.TensorComplexDouble.allocate(ec, ZA, ZB, ZC)

    fill_double_tensor(A, 0)
    fill_double_tensor(B, 1)
    fill_double_tensor(D, 2)
    fill_complex_tensor(ZA, 0)
    fill_complex_tensor(ZB, 1)

    run_scheduler_double(ec, i, j, k, A, B, D, C)
    run_opmin_double(ec, i, j, k, A, B, D, R)
    run_scheduler_complex(ec, i, j, k, ZA, ZB, ZC)

    ec.flush_and_sync()

    pt.write_to_disk(C, prefix + "_sched_d", True, False, 0)
    pt.write_to_disk(R, prefix + "_opmin_d", True, False, 0)
    pt.write_to_disk(ZC, prefix + "_sched_z", True, False, 0)

    ec.flush_and_sync()
    pg.barrier()

    if ec.pg_rank() == 0:
        print(f"Wrote {prefix}_sched_d, {prefix}_opmin_d, {prefix}_sched_z")

    for t in (ZC, ZB, ZA, R, C, D, B, A):
        t.deallocate()


def main():
    args = parse_args()
    pt.initialize(sys.argv, False)

    try:
        run(args.prefix)
    finally:
        pt.finalize()


if __name__ == "__main__":
    main()
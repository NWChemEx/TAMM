# Verifier for the binding-equivalence check.
#
# Reads the on-disk results from Test_Binding_Equivalence.cpp (`cpp_tamm_equiv`)
# and Test_Binding_Equivalence.py (`py_tamm_equiv`) and asserts they are
# numerically identical, proving the pytamm bindings match core C++ TAMM.
# Run after both producers (see Test_Binding_Equivalence.py for the full flow).
import argparse
import itertools
import os
import sys

import pytamm as pt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpp-prefix", default="cpp_tamm_equiv")
    p.add_argument("--py-prefix", default="py_tamm_equiv")
    args, _ = p.parse_known_args()
    return args


def nz_c(b):
    x, y = int(b[0]), int(b[1])
    return x <= y or ((x + y) % 2 == 0)


def make_spaces():
    ispace = pt.IndexSpace(pt.range(6))
    return pt.TiledIndexSpace(ispace, 2)


def make_output_tensors(x):
    info_c = pt.TensorInfo([x, x], nz_c)

    cpp_sched_d = pt.TensorDouble([x, x], info_c)
    py_sched_d = pt.TensorDouble([x, x], info_c)

    cpp_opmin_d = pt.TensorDouble([x, x], info_c)
    py_opmin_d = pt.TensorDouble([x, x], info_c)

    cpp_sched_z = pt.TensorComplexDouble([x, x], info_c)
    py_sched_z = pt.TensorComplexDouble([x, x], info_c)

    return cpp_sched_d, py_sched_d, cpp_opmin_d, py_opmin_d, cpp_sched_z, py_sched_z


def scalar_equal(a, b):
    if isinstance(a, complex) or isinstance(b, complex):
        ca = complex(a)
        cb = complex(b)
        return ca.real == cb.real and ca.imag == cb.imag

    return a == b

# The three on-disk artifacts each producer writes (see Test_Binding_Equivalence.py).
RESULT_SUFFIXES = ("_sched_d", "_opmin_d", "_sched_z")

def missing_inputs(cpp_prefix, py_prefix):
    expected = [p + s for p in (cpp_prefix, py_prefix) for s in RESULT_SUFFIXES]
    return [f for f in expected if not os.path.exists(f)]

def compare_tensor(name, lhs, rhs, ntiles):
    checked = 0

    for bid in itertools.product(range(ntiles), repeat=2):
        lhs_nz = lhs.is_non_zero(bid)
        rhs_nz = rhs.is_non_zero(bid)

        if lhs_nz != rhs_nz:
            raise AssertionError(f"{name}: sparse-structure mismatch at block {bid}")

        if not lhs_nz:
            continue

        n_l = int(lhs.block_size(bid))
        n_r = int(rhs.block_size(bid))

        if n_l != n_r:
            raise AssertionError(
                f"{name}: block-size mismatch at block {bid}: {n_l} != {n_r}"
            )

        lbuf = [0] * n_l
        rbuf = [0] * n_r

        lhs.get(bid, lbuf)
        rhs.get(bid, rbuf)

        for off, (a, b) in enumerate(zip(lbuf, rbuf)):
            if not scalar_equal(a, b):
                raise AssertionError(
                    f"{name}: exact mismatch at block {bid}, offset {off}: {a!r} != {b!r}"
                )

        checked += n_l

    return checked


def run(args):
    pg = pt.ProcGroup.create_world_coll()
    ec = pt.ExecutionContext(pg, pt.DistributionKind.nw, pt.MemoryManagerKind.ga)

    cpp_prefix = args.cpp_prefix
    py_prefix = args.py_prefix 

    # Check if the pre-requisite files from the producers exist
    missing = missing_inputs(args.cpp_prefix, args.py_prefix)
    if missing and ec.pg_rank() == 0:
        sys.stderr.write(
            "Compare_Binding_Equivalence: missing input file(s):\n"
            + "".join(f"  {f}\n" for f in missing)
            + "Run Test_Binding_Equivalence.cpp and Test_Binding_Equivalence.py "
            "first.\n"
        )
        sys.exit(1)

    x = make_spaces()
    ntiles = len(x)

    (
        cpp_sched_d,
        py_sched_d,
        cpp_opmin_d,
        py_opmin_d,
        cpp_sched_z,
        py_sched_z,
    ) = make_output_tensors(x)

    pt.TensorDouble.allocate(ec, cpp_sched_d, py_sched_d, cpp_opmin_d, py_opmin_d)
    pt.TensorComplexDouble.allocate(ec, cpp_sched_z, py_sched_z)

    pt.read_from_disk(cpp_sched_d, cpp_prefix + "_sched_d", True)
    pt.read_from_disk(py_sched_d, py_prefix + "_sched_d", True)

    pt.read_from_disk(cpp_opmin_d, cpp_prefix + "_opmin_d", True)
    pt.read_from_disk(py_opmin_d, py_prefix + "_opmin_d", True)

    pt.read_from_disk(cpp_sched_z, cpp_prefix + "_sched_z", True)
    pt.read_from_disk(py_sched_z, py_prefix + "_sched_z", True)

    ec.flush_and_sync()

    n1 = compare_tensor("double scheduler", cpp_sched_d, py_sched_d, ntiles)
    n2 = compare_tensor("double opmin", cpp_opmin_d, py_opmin_d, ntiles)
    n3 = compare_tensor("complex scheduler", cpp_sched_z, py_sched_z, ntiles)

    if ec.pg_rank() == 0:
        print("EXACT MATCH")
        print(f"  double scheduler scalars checked: {n1}")
        print(f"  double opmin scalars checked:     {n2}")
        print(f"  complex scheduler scalars checked:{n3}")

    for t in (
        py_sched_z,
        cpp_sched_z,
        py_opmin_d,
        cpp_opmin_d,
        py_sched_d,
        cpp_sched_d,
    ):
        t.deallocate()


def main():
    args = parse_args()
    pt.initialize(sys.argv, False)

    try:
        run(args)
    finally:
        pt.finalize()


if __name__ == "__main__":
    main()
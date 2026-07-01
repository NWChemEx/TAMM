#!/usr/bin/env python3

import sys
import pytamm as tamm


def test_tensor_allocate(sch):
    tis = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(10)),
        10,
    )

    tis2 = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(20)),
        20,
    )

    tis3 = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(100)),
        20,
    )

    A = tamm.TensorDouble([tis, tis, tis, tis])
    B = tamm.TensorDouble([tis, tis, tis, tis])
    C = tamm.TensorDouble([tis, tis, tis, tis2])
    D = tamm.TensorDouble([tis, tis, tis, tis2])
    E = tamm.TensorDouble([tis, tis, tis, tis3])

    sch.allocate(A, B, C).deallocate(B, C).execute()
    sch.allocate(D, E).execute()

    tamm.print_memory_usage(int(sch.ec().pg().rank()))


def main():
    tamm.initialize(sys.argv, False)

    try:
        pg = tamm.ProcGroup.create_world_coll()

        ec = tamm.ExecutionContext(
            pg,
            tamm.DistributionKind.nw,
            tamm.MemoryManagerKind.ga,
        )

        sch = tamm.Scheduler(ec)

        test_tensor_allocate(sch)

    finally:
        tamm.finalize(True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import sys
import pytamm as tamm


def check_value(obj, val, tol=1.0e-10):
    if isinstance(obj, (tamm.LabeledTensorDouble, tamm.LabeledTensorComplexDouble)):
        lt = obj
    else:
        lt = obj()

    tensor = lt.tensor()

    for itval in tamm.LabelLoopNest(lt.labels()):
        blockid = tamm.translate_blockid(itval, lt)
        size = int(tensor.block_size(blockid))
        buf = [0.0] * size
        tensor.get(blockid, buf)

        for x in buf:
            if isinstance(x, complex):
                diff = abs(x - val)
            else:
                diff = abs(float(x) - float(val))

            if diff >= tol:
                return False

    return True


def test_unit_tiled_view_tensor(ec, size, tile_size):
    IS = tamm.IndexSpace(
        tamm.range(size),
        {
            "occ": [tamm.range(0, size // 2)],
            "virt": [tamm.range(size // 2, size)],
        },
    )

    TIS = tamm.TiledIndexSpace(IS, int(tile_size))

    X, Y, Z = TIS.labels("all", count=3)

    T_full = tamm.TensorDouble([X, Y, Z])
    T_copy = tamm.TensorDouble([X, Y, Z])

    sch = tamm.Scheduler(ec)

    T_full.allocate_self(ec)
    T_copy.allocate_self(ec)

    tamm.random_ip(T_full)

    sch(T_copy(X, Y, Z), "=", T_full(X, Y, Z)).execute()

    # Tensor<double> T_unit_1{T_full, 1};
    T_unit_1 = tamm.TensorDouble(T_full, 1)

    # Both memory regions should be the same
    assert T_full.same_memory_region(T_unit_1)

    # Different distributions for unit tiled view and normal tensor
    assert not T_full.same_distribution(T_unit_1)

    # Unit tiled view tensor should already be allocated
    assert T_unit_1.is_allocated()

    # Test two allocates doesn't break things
    T_unit_1.allocate_force(ec)

    # -------------------------------------------------------------------------
    # SetOp tests
    # -------------------------------------------------------------------------
    sch(T_full("X", "Y", "Z"), "=", 42.0).execute()

    assert check_value(T_full, 42.0)
    assert check_value(T_unit_1, 42.0)

    sch(T_full("X", "Y", "Z"), "=", T_copy("X", "Y", "Z")).execute()

    sch(T_unit_1("x", "Y", "Z"), "=", 42.0).execute()

    assert check_value(T_unit_1, 42.0)
    assert check_value(T_full, 42.0)

    sch(T_full("X", "Y", "Z"), "=", T_copy("X", "Y", "Z")).execute()

    # -------------------------------------------------------------------------
    # AddOp tests
    # -------------------------------------------------------------------------
    A_full = tamm.TensorDouble([X, Y, Z])
    A_full.allocate_self(ec)

    tamm.random_ip(A_full)

    A_unit_1 = tamm.TensorDouble(A_full, 1)

    sch(T_full("X", "Y", "Z"), "+=", A_full("X", "Y", "Z")).execute()

    sch(T_full("X", "Y", "Z"), "=", T_copy("X", "Y", "Z")).execute()

    sch(T_unit_1("x", "Y", "Z"), "+=", A_unit_1("x", "Y", "Z")).execute()

    sch(T_full("X", "Y", "Z"), "=", T_copy("X", "Y", "Z")).execute()

    # -------------------------------------------------------------------------
    # MultOp tests
    # -------------------------------------------------------------------------
    B_full = tamm.TensorDouble([X, Y])
    B_full.allocate_self(ec)

    tamm.random_ip(B_full)

    B_unit_1 = tamm.TensorDouble(B_full, 1)
    _ = B_unit_1

    sch(
        T_full("X", "Y", "Z"),
        "=",
        A_full("X", "Y", "V") * B_full("V", "Z"),
    ).execute()

    sch(T_full("X", "Y", "Z"), "=", T_copy("X", "Y", "Z")).execute()

    sch(
        T_unit_1("x", "Y", "Z"),
        "=",
        A_unit_1("x", "Y", "V") * B_full("V", "Z"),
    ).execute()

    # -------------------------------------------------------------------------
    # Sliced update
    # -------------------------------------------------------------------------
    unit_tis_1 = tamm.TiledIndexSpace(
        T_unit_1.tiled_index_spaces()[0],
        tamm.range(2, 3),
    )

    x_2 = unit_tis_1.label()

    sch(T_unit_1(x_2, Y, Z), "=", 1.0).execute()

    T_unit_2 = tamm.TensorDouble(T_full, 2)

    unit_tis_2 = tamm.TiledIndexSpace(
        T_unit_2.tiled_index_spaces()[1],
        tamm.range(2, 3),
    )

    y_2 = unit_tis_2.label()

    sch(T_unit_2(x_2, y_2, Z), "=", 2.0).execute()

    print(
        "\n"
        + "*" * 25
        + "\nRunning MO space tests!\n"
        + "*" * 25
        + "\n"
    )

    # -------------------------------------------------------------------------
    # MO space tests
    # -------------------------------------------------------------------------
    MO_IS = tamm.IndexSpace(
        tamm.range(0, 50),
        {
            "occ": [tamm.range(0, 15)],
            "virt": [tamm.range(15, 50)],
        },
    )

    mo_tiles = [10, 5, 10, 10, 10, 5]
    MO = tamm.TiledIndexSpace(MO_IS, mo_tiles)

    h1, h2, h3 = MO.labels("occ", count=3)

    t1 = tamm.TensorDouble([h1, h2])
    t2 = tamm.TensorDouble([h1, h2])
    tmp = tamm.TensorDouble([h1])

    sch.allocate(t1, t2, tmp).execute()

    t1_ut = tamm.TensorDouble(t1, 1)

    t1_utis = tamm.TiledIndexSpace(
        t1_ut.tiled_index_spaces()[0],
        tamm.range(2, 3),
    )

    t1_ut_l1 = t1_utis.label()

    sch(tmp(h3), "=", t1_ut(t1_ut_l1, h2) * t2(h2, h3)).execute()

    # -------------------------------------------------------------------------
    # AO unit-tiled scalar extraction loop
    # -------------------------------------------------------------------------
    ao_tiles = [1, 3]
    AO = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(4)),
        ao_tiles,
    )

    T = tamm.TensorDouble([AO, AO])

    sch.allocate(T).execute()

    tamm.random_ip(T)

    T_ut = tamm.TensorDouble(T, 2)

    tmp2 = tamm.TensorDouble()

    sch.allocate(tmp2).execute()

    for ii in range(4):
        for jj in range(4):
            tis1 = tamm.TiledIndexSpace(
                T_ut.tiled_index_spaces()[0],
                tamm.range(ii, ii + 1),
            )

            tis2 = tamm.TiledIndexSpace(
                T_ut.tiled_index_spaces()[1],
                tamm.range(jj, jj + 1),
            )

            l1 = tis1.label()
            l2 = tis2.label()

            sch(tmp2(), "=", T_ut(l1, l2)).execute()

            val = tamm.get_scalar(tmp2)

            if int(ec.pg().rank()) == 0:
                print(ii, jj, val)

    print("Finished tests!")


def main():
    tamm.initialize(sys.argv, False)

    try:
        pg = tamm.ProcGroup.create_world_coll()

        # The C++ test uses the low-level ExecutionContext constructor with
        # Distribution_NW, MemoryManagerGA, and RuntimeEngine. The exposed Python
        # binding equivalent is the standard nw/ga constructor.
        ec = tamm.ExecutionContext(
            pg,
            tamm.DistributionKind.nw,
            tamm.MemoryManagerKind.ga,
        )

        test_unit_tiled_view_tensor(ec, 20, 5)

    finally:
        tamm.finalize(True)


if __name__ == "__main__":
    main()
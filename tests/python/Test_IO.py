#!/usr/bin/env python3

import re
import sys
import pytamm as tamm

T = float
tammio = True
profileio = True
init_value = 21.0


def atoi_cpp(value):
    s = str(value)
    m = re.match(r"^\s*([+-]?\d+)", s)
    if not m:
        return 0
    return int(m.group(1))


def world_rank():
    return int(tamm.ProcGroup.world_rank())


def io_stats(gec, tensor):
    rank = int(gec.pg().rank())

    nelements = 1.0

    # Heuristic: Use 1 agg for every 14 GiB
    ne_mb = 131072 * 14.0

    ndims = int(tensor.num_modes())
    tiss = tensor.tiled_index_spaces()

    for i in range(ndims):
        nelements *= int(tiss[i].index_space().num_indices())

    nagg = int(nelements / (ne_mb * 1024) + 1)
    nppn = f"{nagg} nodes"

    if rank == 0 and profileio:
        size_gib = (nelements * 8.0) / (1024 * 1024 * 1024.0)
        print(
            f"tensor size: {size_gib:.2f}GiB, "
            f"can write to disk using upto: {nppn}"
        )


def setupTIS(noa, nva):
    n_occ_alpha = noa
    n_occ_beta = noa
    freeze_core = 0
    freeze_virtual = 0

    nbf = noa + nva
    nmo = noa * 2 + nva * 2

    n_vir_alpha = nva
    n_vir_beta = nva

    nocc = n_occ_alpha * 2

    sizes = [
        n_occ_alpha - freeze_core,
        n_occ_beta - freeze_core,
        n_vir_alpha - freeze_virtual,
        n_vir_beta - freeze_virtual,
    ]
    _ = sizes

    total_orbitals = nmo - 2 * freeze_core - 2 * freeze_virtual

    MO_IS = tamm.IndexSpace(
        tamm.range(0, total_orbitals),
        {
            "occ": [tamm.range(0, nocc)],
            "occ_alpha": [tamm.range(0, n_occ_alpha)],
            "occ_beta": [tamm.range(n_occ_alpha, nocc)],
            "virt": [tamm.range(nocc, total_orbitals)],
            "virt_alpha": [tamm.range(nocc, nocc + n_vir_alpha)],
            "virt_beta": [tamm.range(nocc + n_vir_alpha, total_orbitals)],
        },
        {
            tamm.Spin(1): [
                tamm.range(0, n_occ_alpha),
                tamm.range(nocc, nocc + n_vir_alpha),
            ],
            tamm.Spin(2): [
                tamm.range(n_occ_alpha, nocc),
                tamm.range(nocc + n_vir_alpha, total_orbitals),
            ],
        },
    )

    tce_tile = int(nbf / 10)

    if tce_tile < 50 or tce_tile > 100:
        if tce_tile < 50:
            tce_tile = 50
        if tce_tile > 100:
            tce_tile = 100

        if world_rank() == 0:
            print()
            print(f"Resetting tilesize to: {tce_tile}")

    if world_rank() == 0:
        print(f"nbf = {nbf}")
        print(f"nmo = {nmo}")
        print(f"nocc = {nocc}")
        print(f"n_occ_alpha = {n_occ_alpha}")
        print(f"n_vir_alpha = {n_vir_alpha}")
        print(f"n_occ_beta = {n_occ_beta}")
        print(f"n_vir_beta = {n_vir_beta}")
        print(f"tilesize   = {tce_tile}")

    mo_tiles = []

    est_nt = n_occ_alpha // tce_tile
    last_tile = n_occ_alpha % tce_tile
    for _ in range(est_nt):
        mo_tiles.append(tce_tile)
    if last_tile > 0:
        mo_tiles.append(last_tile)

    est_nt = n_occ_beta // tce_tile
    last_tile = n_occ_beta % tce_tile
    for _ in range(est_nt):
        mo_tiles.append(tce_tile)
    if last_tile > 0:
        mo_tiles.append(last_tile)

    est_nt = n_vir_alpha // tce_tile
    last_tile = n_vir_alpha % tce_tile
    for _ in range(est_nt):
        mo_tiles.append(tce_tile)
    if last_tile > 0:
        mo_tiles.append(last_tile)

    est_nt = n_vir_beta // tce_tile
    last_tile = n_vir_beta % tce_tile
    for _ in range(est_nt):
        mo_tiles.append(tce_tile)
    if last_tile > 0:
        mo_tiles.append(last_tile)

    MO = tamm.TiledIndexSpace(MO_IS, mo_tiles)
    tis_i = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(6 * nva)),
        tce_tile,
    )

    return MO, tis_i, total_orbitals


def read_write(tensor, tstring):
    hdf5_str = tstring + "_hdf5"
    mpiio_str = tstring + "_mpiio"
    _ = mpiio_str

    tamm.write_to_disk(tensor, hdf5_str, tammio, profileio)
    tamm.read_from_disk(tensor, hdf5_str, tammio, None, profileio)


def test_io_2d(sch, tis, tis_i, tensor_cls=tamm.TensorDouble):
    N = tis("all")
    O = tis("occ")
    V = tis("virt")
    K = tis_i("all")
    _ = (N, K)

    t2_oo = tensor_cls([O, O])
    t2_ov = tensor_cls([O, V])
    t2_vv = tensor_cls([V, V])

    sch.allocate(t2_oo, t2_ov, t2_vv)
    sch(t2_oo(), "=", init_value)
    sch(t2_ov(), "=", init_value)
    sch(t2_vv(), "=", init_value)
    sch.execute()

    read_write(t2_oo, "t2_oo")
    read_write(t2_ov, "t2_ov")
    read_write(t2_vv, "t2_vv")

    sch.deallocate(t2_oo, t2_ov, t2_vv).execute()


def test_io_3d(sch, tis, tis_i, tensor_cls=tamm.TensorDouble):
    N = tis("all")
    O = tis("occ")
    V = tis("virt")
    K = tis_i("all")
    _ = N

    t3_ook = tensor_cls([O, O, K])
    t3_ovk = tensor_cls([O, V, K])
    t3_vvk = tensor_cls([V, V, K])

    t3_ooo = tensor_cls([O, O, O])
    t3_oov = tensor_cls([O, O, V])
    t3_ovv = tensor_cls([O, V, V])
    t3_vvv = tensor_cls([V, V, V])

    sch.allocate(t3_ook, t3_ovk, t3_vvk)
    sch.allocate(t3_ooo, t3_oov, t3_ovv, t3_vvv)

    sch(t3_ook(), "=", init_value)
    sch(t3_ovk(), "=", init_value)
    sch(t3_vvk(), "=", init_value)
    sch(t3_ooo(), "=", init_value)
    sch(t3_oov(), "=", init_value)
    sch(t3_ovv(), "=", init_value)
    sch(t3_vvv(), "=", init_value)

    sch.execute()

    read_write(t3_ook, "t3_ook")
    read_write(t3_ovk, "t3_ovk")
    read_write(t3_vvk, "t3_vvk")
    read_write(t3_ooo, "t3_ooo")
    read_write(t3_oov, "t3_oov")
    read_write(t3_ovv, "t3_ovv")
    read_write(t3_vvv, "t3_vvv")

    sch.deallocate(t3_ook, t3_ovk, t3_vvk).execute()
    sch.deallocate(t3_ooo, t3_oov, t3_ovv, t3_vvv).execute()


def test_io_4d(sch, tis, tis_i, tensor_cls=tamm.TensorDouble):
    N = tis("all")
    O = tis("occ")
    V = tis("virt")
    K = tis_i("all")
    _ = (N, K)

    t_oooo = tensor_cls([O, O, O, O], [2, 2])
    t_ooov = tensor_cls([O, O, O, V], [2, 2])
    t_oovv = tensor_cls([O, O, V, V], [2, 2])
    t_ovvv = tensor_cls([O, V, V, V], [2, 2])

    sch.allocate(t_oooo)
    sch(t_oooo(), "=", init_value)
    sch.execute()

    sch.allocate(t_ooov)
    sch(t_ooov(), "=", init_value)
    sch.execute()

    sch.allocate(t_oovv)
    sch(t_oovv(), "=", init_value)
    sch.execute()

    sch.allocate(t_ovvv)
    sch(t_ovvv(), "=", init_value)
    sch.execute()

    read_write(t_oooo, "t_oooo")
    read_write(t_ooov, "t_ooov")
    read_write(t_oovv, "t_oovv")
    read_write(t_ovvv, "t_ovvv")

    sch.deallocate(t_oooo).execute()
    sch.deallocate(t_ooov).execute()
    sch.deallocate(t_oovv).execute()
    sch.deallocate(t_ovvv).execute()


def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        print("Please provide a dimension size!")
        return 0

    tamm.initialize(argv, False)

    try:
        pg = tamm.ProcGroup.create_world_coll()

        ec = tamm.ExecutionContext(
            pg,
            tamm.DistributionKind.nw,
            tamm.MemoryManagerKind.ga,
        )

        dense_pg = ec.pg()
        ec_dense = tamm.ExecutionContext(
            dense_pg,
            tamm.DistributionKind.dense,
            tamm.MemoryManagerKind.ga,
        )

        sch = tamm.Scheduler(ec_dense)

        nbf = atoi_cpp(argv[1])
        ts_ = max(30, int(nbf * 0.05))

        # C++ reference has these commented out:
        #
        # TIS, TIS_I, total_orbitals = setupTIS(nbf, ts_)
        # test_io_2d(sch, TIS, TIS_I)
        # test_io_3d(sch, TIS, TIS_I)
        # test_io_4d(sch, TIS, TIS_I)

        gc_tiles = []

        est_nt = nbf // ts_
        last_tile = nbf % ts_

        for _ in range(est_nt):
            gc_tiles.append(ts_)

        if last_tile > 0:
            gc_tiles.append(last_tile)

        tc_ij = tamm.TiledIndexSpace(
            tamm.IndexSpace(tamm.range(nbf)),
            gc_tiles,
        )

        tci = tamm.TiledIndexSpace(
            tamm.IndexSpace(tamm.range(12 * nbf)),
            12 * nbf,
        )

        gc_tensor = tamm.TensorDouble([tc_ij, tc_ij, tci])
        gc_tensor.set_dense()

        io_stats(ec_dense, gc_tensor)

        sch.allocate(gc_tensor).execute()

        if ec.print():
            print("Writing a 3D tensor of size (NxNx12N) to disk ... ")

        tamm.write_to_disk(gc_tensor, "tensor3d", True, True)

        sch.deallocate(gc_tensor).execute()

    finally:
        tamm.finalize(True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
import pytamm as tamm


def construct_full_dependency_2(in_tis1, in_tis2, out_tis):
    result = {}

    for tile_1 in in_tis1:
        for tile_2 in in_tis2:
            result[(int(tile_1), int(tile_2))] = out_tis

    return result


def dlpno_T1_T2_allocate(sch, N, tilesize):
    LMO = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(N)),
        tilesize,
    )

    depMO = construct_full_dependency_2(LMO, LMO, LMO)

    PNO = tamm.TiledIndexSpace(
        LMO,
        [LMO, LMO],
        depMO,
    )

    i, j = LMO.labels("all", count=2)
    a, e = PNO.labels("all", count=2)

    t1 = tamm.TensorDouble([i, e(i, i)])
    t2 = tamm.TensorDouble([i, j, a(i, j), e(i, j)])

    sch.allocate(t1, t2).execute()

    return t1, t2


def main():
    tamm.initialize(["dlpno_T1_T2_allocate.py"], False)

    try:
        pg = tamm.ProcGroup.create_world_coll()
        ec = tamm.ExecutionContext(
            pg,
            tamm.DistributionKind.nw,
            tamm.MemoryManagerKind.ga,
        )

        sch = tamm.Scheduler(ec)

        dlpno_T1_T2_allocate(sch, 10, 2)

    finally:
        tamm.finalize(True)


if __name__ == "__main__":
    main()
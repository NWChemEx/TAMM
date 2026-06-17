#!/usr/bin/env python3

import json
import pathlib
import sys

import pytamm as tamm


def L(lt):
    return tamm.as_op(lt)


def make_ltop(tensor, *labels):
    return L(tensor(*labels))


def set_labeled_expr(lhs, expr):
    # C++ uses:
    #   lhs.set(expr);
    #
    # If your binding exposes set(), use it. The binding shown earlier exposes
    # update(), so this falls back to update().
    if hasattr(lhs, "set"):
        lhs.set(expr)
    else:
        lhs.update(expr)


def print_debug_binarized(symbol_table, lhs_lt, expr, use_opmin):
    op_to_print = expr.clone()

    if use_opmin:
        opmin = tamm.OpMin(symbol_table)
        op_to_print = opmin.optimize_all(lhs_lt, op_to_print, True)

    op_cost = tamm.OpCostCalculator(symbol_table)
    op_cost.print_op_binarized(lhs_lt, op_to_print)


def dlpno_test(params):
    # -------------------------------------------------------------------------
    # Get dimension sizes and tile sizes from input JSON
    # -------------------------------------------------------------------------
    dim_sizes = params["dim_sizes"]

    ao_size = int(dim_sizes["AO"])
    mo_size = int(dim_sizes["MO"])
    occ_size = int(dim_sizes["MO_occ"])
    virt_size = int(dim_sizes["MO_virt"])

    if (occ_size + virt_size) != mo_size:
        raise RuntimeError(
            f"[TAMM ERROR] MO({mo_size}) size should be equal "
            f"to Virt({virt_size}) + Occ({occ_size})!"
        )

    df_size = int(dim_sizes["DF"])
    lmop_size = int(dim_sizes["LMOP"])
    pao_size = int(dim_sizes["PAO"])
    pno_size = int(dim_sizes["PNO"])

    tile_size = int(params["tile_size"])
    use_opmin = bool(params["use_opmin"])
    do_profile = bool(params["do_profile"])
    ex_hw = tamm.ExecutionHW.GPU if bool(params["use_gpu"]) else tamm.ExecutionHW.CPU
    print_debug_info = bool(params["print_debug_info"])

    # -------------------------------------------------------------------------
    # Initialize TiledIndexSpaces
    # -------------------------------------------------------------------------
    AO = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(ao_size)),
        tile_size,
    )

    MO = tamm.TiledIndexSpace(
        tamm.IndexSpace(
            tamm.range(mo_size),
            {
                "occ": [tamm.range(occ_size)],
                "virt": [tamm.range(occ_size, mo_size)],
            },
        ),
        tile_size,
    )

    DF = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(df_size)),
        tile_size,
    )

    LMOP = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(lmop_size)),
        tile_size,
    )

    PAO = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(pao_size)),
        tile_size,
    )

    PNO = tamm.TiledIndexSpace(
        tamm.IndexSpace(tamm.range(pno_size)),
        tile_size,
    )

    # -------------------------------------------------------------------------
    # Constructing EC and Scheduler
    # -------------------------------------------------------------------------
    pg = tamm.ProcGroup.create_world_coll()

    ec = tamm.ExecutionContext(
        pg,
        tamm.DistributionKind.nw,
        tamm.MemoryManagerKind.ga,
    )

    rank = int(pg.rank())
    sch = tamm.Scheduler(ec)

    # -------------------------------------------------------------------------
    # Constructs for new execution
    # -------------------------------------------------------------------------
    symbol_table = tamm.SymbolTable()
    op_executor = tamm.OpExecutor(sch, symbol_table)

    # -------------------------------------------------------------------------
    # Amplitude tensors
    # -------------------------------------------------------------------------
    dT1 = tamm.TensorDouble([PNO, LMOP])
    dT2 = tamm.TensorDouble([PNO, PNO, LMOP])
    dr1 = tamm.TensorDouble([PNO, LMOP])
    dr2 = tamm.TensorDouble([PNO, PNO, LMOP])

    # -------------------------------------------------------------------------
    # 3 Center Integrals
    # -------------------------------------------------------------------------
    dTEoo = tamm.TensorDouble([LMOP, LMOP, DF])
    dTEov = tamm.TensorDouble([LMOP, PAO, DF])
    dTEvv = tamm.TensorDouble([PAO, PAO, DF])

    # -------------------------------------------------------------------------
    # Transformation tensors
    # -------------------------------------------------------------------------
    d = tamm.TensorDouble([LMOP, PAO, PNO])
    Sijkl = tamm.TensorDouble([LMOP, LMOP, PNO, PNO])

    # -------------------------------------------------------------------------
    # Expand O to LMOP
    # -------------------------------------------------------------------------
    expand = tamm.TensorDouble([MO("occ"), LMOP])

    # -------------------------------------------------------------------------
    # Register tensor names to the symbol table
    # -------------------------------------------------------------------------
    tamm.register_symbols(
        symbol_table,
        dT1=dT1,
        dT2=dT2,
        dr1=dr1,
        dr2=dr2,
        dTEoo=dTEoo,
        dTEov=dTEov,
        dTEvv=dTEvv,
        d=d,
        Sijkl=Sijkl,
        expand=expand,
    )

    # -------------------------------------------------------------------------
    # Allocate all tensors
    # -------------------------------------------------------------------------
    sch.allocate(dT1, dT2, dr1, dr2, dTEoo, dTEov, dTEvv, d, Sijkl, expand)
    sch(dr1(), "=", 0.0)
    sch(dr2(), "=", 0.0)
    sch.execute()

    if rank == 0:
        print("Allocated all tensors.")

    # -------------------------------------------------------------------------
    # Initialize tensors with random values
    # -------------------------------------------------------------------------
    tamm.random_ip(dT1)
    tamm.random_ip(dT2)
    tamm.random_ip(dTEoo)
    tamm.random_ip(dTEov)
    tamm.random_ip(dTEvv)
    tamm.random_ip(d)
    tamm.random_ip(Sijkl)
    tamm.random_ip(expand)

    if rank == 0:
        print("Initialized all input tensors with random values.")

    E = make_ltop

    # -------------------------------------------------------------------------
    # Most expensive doubles
    # -------------------------------------------------------------------------
    doubles_5 = (
        E(dTEov, "mn", "e_mu", "K")
        * E(dTEov, "mn", "f_mu", "K")
        * E(dT1, "e_ii", "ii")
        * E(dT1, "f_jj", "jj")
        * E(dT2, "a_mn", "b_mn", "mn")
        * E(d, "ii", "e_mu", "e_ii")
        * E(d, "jj", "f_mu", "f_jj")
        * E(Sijkl, "mn", "ij", "a_mn", "a_ij")
        * E(Sijkl, "mn", "ij", "b_mn", "b_ij")
        * E(expand, "j", "jj")
        * E(expand, "i", "ii")
        * E(expand, "i", "ij")
        * E(expand, "j", "ij")
    )

    doubles_10 = (
        E(dTEov, "mm", "e_mu", "K")
        * E(dTEov, "nn", "f_mu", "K")
        * E(dT1, "e_ii", "ii")
        * E(dT1, "f_jj", "jj")
        * E(dT1, "a_mm", "mm")
        * E(dT1, "b_nn", "nn")
        * E(d, "ii", "e_mu", "e_ii")
        * E(d, "jj", "f_mu", "f_jj")
        * E(Sijkl, "nn", "ij", "b_nn", "b_ij")
        * E(Sijkl, "mm", "ij", "a_mm", "a_ij")
        * E(expand, "j", "jj")
        * E(expand, "i", "ii")
        * E(expand, "i", "ij")
        * E(expand, "j", "ij")
    )

    doubles_15 = (
        -1.0
        * E(dTEvv, "a_mu", "e_mu", "K")
        * E(dTEov, "mm", "f_mu", "K")
        * E(dT1, "b_mm", "mm")
        * E(dT1, "e_ii", "ii")
        * E(dT1, "f_jj", "jj")
        * E(d, "ii", "e_mu", "e_ii")
        * E(d, "jj", "f_mu", "f_jj")
        * E(d, "ij", "a_mu", "a_ij")
        * E(Sijkl, "mm", "ij", "b_mm", "b_ij")
        * E(expand, "j", "jj")
        * E(expand, "i", "ii")
        * E(expand, "i", "ij")
        * E(expand, "j", "ij")
    )

    doubles_16 = (
        -1.0
        * E(dTEov, "mm", "e_mu", "K")
        * E(dTEvv, "b_mu", "f_mu", "K")
        * E(dT1, "a_mm", "mm")
        * E(dT1, "e_ii", "ii")
        * E(dT1, "f_jj", "jj")
        * E(d, "ii", "e_mu", "e_ii")
        * E(d, "jj", "f_mu", "f_jj")
        * E(d, "ij", "b_mu", "b_ij")
        * E(Sijkl, "mm", "ij", "a_mm", "a_ij")
        * E(expand, "j", "jj")
        * E(expand, "i", "ii")
        * E(expand, "i", "ij")
        * E(expand, "j", "ij")
    )

    doubles_37 = (
        -2.0
        * E(dTEov, "mj", "e_mu", "K")
        * E(dTEov, "nn", "f_mu", "K")
        * E(dT1, "e_ii", "ii")
        * E(dT1, "f_nn", "nn")
        * E(dT2, "a_mj", "b_mj", "mj")
        * E(d, "ii", "e_mu", "e_ii")
        * E(d, "nn", "f_mu", "f_nn")
        * E(Sijkl, "mj", "ij", "b_mj", "b_ij")
        * E(Sijkl, "mj", "ij", "a_mj", "a_ij")
        * E(expand, "i", "ii")
        * E(expand, "i", "ij")
        * E(expand, "j", "mj")
        * E(expand, "j", "ij")
    )

    doubles_38 = (
        -2.0
        * E(dTEov, "mi", "e_mu", "K")
        * E(dTEov, "nn", "f_mu", "K")
        * E(dT1, "e_jj", "jj")
        * E(dT1, "f_nn", "nn")
        * E(dT2, "b_mi", "a_mi", "mi")
        * E(d, "jj", "e_mu", "e_jj")
        * E(d, "nn", "f_mu", "f_nn")
        * E(Sijkl, "mi", "ij", "b_mi", "b_ij")
        * E(Sijkl, "mi", "ij", "a_mi", "a_ij")
        * E(expand, "j", "ij")
        * E(expand, "i", "mi")
        * E(expand, "i", "ij")
        * E(expand, "j", "jj")
    )

    doubles_41 = (
        E(dTEov, "mj", "f_mu", "K")
        * E(dTEov, "nn", "e_mu", "K")
        * E(dT1, "e_ii", "ii")
        * E(dT1, "f_nn", "nn")
        * E(dT2, "a_mj", "b_mj", "mj")
        * E(d, "ii", "e_mu", "e_ii")
        * E(d, "nn", "f_mu", "f_nn")
        * E(Sijkl, "mj", "ij", "b_mj", "b_ij")
        * E(Sijkl, "mj", "ij", "a_mj", "a_ij")
        * E(expand, "i", "ii")
        * E(expand, "i", "ij")
        * E(expand, "j", "mj")
        * E(expand, "j", "ij")
    )

    doubles_42 = (
        E(dTEov, "mi", "f_mu", "K")
        * E(dTEov, "nn", "e_mu", "K")
        * E(dT1, "e_jj", "jj")
        * E(dT1, "f_nn", "nn")
        * E(dT2, "b_mi", "a_mi", "mi")
        * E(d, "jj", "e_mu", "e_jj")
        * E(d, "nn", "f_mu", "f_nn")
        * E(Sijkl, "mi", "ij", "b_mi", "b_ij")
        * E(Sijkl, "mi", "ij", "a_mi", "a_ij")
        * E(expand, "j", "ij")
        * E(expand, "i", "mi")
        * E(expand, "i", "ij")
        * E(expand, "j", "jj")
    )

    doubles_65 = (
        -2.0
        * E(dTEov, "mj", "e_mu", "K")
        * E(dTEov, "nn", "f_mu", "K")
        * E(dT1, "f_ii", "ii")
        * E(dT1, "a_nn", "nn")
        * E(dT2, "e_mj", "b_mj", "mj")
        * E(d, "mj", "e_mu", "e_mj")
        * E(d, "ii", "f_mu", "f_ii")
        * E(Sijkl, "mj", "ij", "b_mj", "b_ij")
        * E(Sijkl, "nn", "ij", "a_nn", "a_ij")
        * E(expand, "i", "ii")
        * E(expand, "i", "ij")
        * E(expand, "j", "mj")
        * E(expand, "j", "ij")
    )

    doubles_66 = (
        -2.0
        * E(dTEov, "mi", "e_mu", "K")
        * E(dTEov, "nn", "f_mu", "K")
        * E(dT1, "f_jj", "jj")
        * E(dT1, "b_nn", "nn")
        * E(dT2, "e_mi", "a_mi", "mi")
        * E(d, "mi", "e_mu", "e_mi")
        * E(d, "jj", "f_mu", "f_jj")
        * E(Sijkl, "mi", "ij", "a_mi", "a_ij")
        * E(Sijkl, "nn", "ij", "b_nn", "b_ij")
        * E(expand, "j", "ij")
        * E(expand, "i", "mi")
        * E(expand, "i", "ij")
        * E(expand, "j", "jj")
    )

    doubles_79 = (
        E(dTEov, "mj", "e_mu", "K")
        * E(dTEov, "nn", "f_mu", "K")
        * E(dT1, "f_ii", "ii")
        * E(dT1, "a_nn", "nn")
        * E(dT2, "b_mj", "e_mj", "mj")
        * E(d, "mj", "e_mu", "e_mj")
        * E(d, "ii", "f_mu", "f_ii")
        * E(Sijkl, "mj", "ij", "b_mj", "b_ij")
        * E(Sijkl, "nn", "ij", "a_nn", "a_ij")
        * E(expand, "i", "ii")
        * E(expand, "i", "ij")
        * E(expand, "j", "mj")
        * E(expand, "j", "ij")
    )

    doubles_80 = (
        E(dTEov, "mi", "e_mu", "K")
        * E(dTEov, "nn", "f_mu", "K")
        * E(dT1, "f_jj", "jj")
        * E(dT1, "b_nn", "nn")
        * E(dT2, "a_mi", "e_mi", "mi")
        * E(d, "mi", "e_mu", "e_mi")
        * E(d, "jj", "f_mu", "f_jj")
        * E(Sijkl, "mi", "ij", "a_mi", "a_ij")
        * E(Sijkl, "nn", "ij", "b_nn", "b_ij")
        * E(expand, "j", "ij")
        * E(expand, "i", "mi")
        * E(expand, "i", "ij")
        * E(expand, "j", "jj")
    )

    doubles_93 = (
        E(dTEov, "mj", "f_mu", "K")
        * E(dTEov, "nn", "e_mu", "K")
        * E(dT1, "f_ii", "ii")
        * E(dT1, "a_nn", "nn")
        * E(dT2, "e_mj", "b_mj", "mj")
        * E(d, "mj", "e_mu", "e_mj")
        * E(d, "ii", "f_mu", "f_ii")
        * E(Sijkl, "mj", "ij", "b_mj", "b_ij")
        * E(Sijkl, "nn", "ij", "a_nn", "a_ij")
        * E(expand, "i", "ii")
        * E(expand, "i", "ij")
        * E(expand, "j", "mj")
        * E(expand, "j", "ij")
    )

    doubles_94 = (
        E(dTEov, "mi", "f_mu", "K")
        * E(dTEov, "nn", "e_mu", "K")
        * E(dT1, "f_jj", "jj")
        * E(dT1, "b_nn", "nn")
        * E(dT2, "e_mi", "a_mi", "mi")
        * E(d, "mi", "e_mu", "e_mi")
        * E(d, "jj", "f_mu", "f_jj")
        * E(Sijkl, "mi", "ij", "a_mi", "a_ij")
        * E(Sijkl, "nn", "ij", "b_nn", "b_ij")
        * E(expand, "j", "ij")
        * E(expand, "i", "mi")
        * E(expand, "i", "ij")
        * E(expand, "j", "jj")
    )

    doubles_103 = (
        -0.5
        * E(dTEov, "mj", "f_mu", "K")
        * E(dTEov, "nn", "e_mu", "K")
        * E(dT1, "f_ii", "ii")
        * E(dT1, "a_nn", "nn")
        * E(dT2, "b_mj", "e_mj", "mj")
        * E(d, "mj", "e_mu", "e_mj")
        * E(d, "ii", "f_mu", "f_ii")
        * E(Sijkl, "mj", "ij", "b_mj", "b_ij")
        * E(Sijkl, "nn", "ij", "a_nn", "a_ij")
        * E(expand, "i", "ii")
        * E(expand, "i", "ij")
        * E(expand, "j", "mj")
        * E(expand, "j", "ij")
    )

    doubles_104 = (
        -0.5
        * E(dTEov, "mi", "f_mu", "K")
        * E(dTEov, "nn", "e_mu", "K")
        * E(dT1, "f_jj", "jj")
        * E(dT1, "b_nn", "nn")
        * E(dT2, "a_mi", "e_mi", "mi")
        * E(d, "mi", "e_mu", "e_mi")
        * E(d, "jj", "f_mu", "f_jj")
        * E(Sijkl, "mi", "ij", "a_mi", "a_ij")
        * E(Sijkl, "nn", "ij", "b_nn", "b_ij")
        * E(expand, "j", "ij")
        * E(expand, "i", "mi")
        * E(expand, "i", "ij")
        * E(expand, "j", "jj")
    )

    # Keep these constructed, matching the C++ source, even though execution of
    # several of them is commented out.
    _ = (
        doubles_16,
        doubles_38,
        doubles_42,
        doubles_66,
        doubles_79,
        doubles_80,
        doubles_94,
        doubles_103,
        doubles_104,
    )

    active_exprs = [
        ("doubles_5", doubles_5),
        ("doubles_10", doubles_10),
        ("doubles_15", doubles_15),
        ("doubles_37", doubles_37),
        ("doubles_41", doubles_41),
        ("doubles_65", doubles_65),
        ("doubles_93", doubles_93),
    ]

    for name, expr in active_exprs:
        lhs = dr2("a_ij", "b_ij", "ij")

        set_labeled_expr(lhs, expr)

        if rank == 0 and print_debug_info:
            print_debug_binarized(symbol_table, lhs, expr, use_opmin)

        op_executor.execute(dr2, use_opmin, ex_hw, do_profile)

        if rank == 0:
            print(f"Finished executing {name}.")


def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        print("Please provide a JSON input file!")
        return 1

    filename = argv[1]
    path = pathlib.Path(filename)

    if path.suffix != ".json":
        print("Only JSON format is supported. Please provide a JSON input file!")
        return 1

    if not path.exists():
        print(f"Input file provided [{filename}] does not exist!")
        return 1

    input_filestem = path.stem
    _ = input_filestem

    with path.open("r") as input_params:
        params = json.load(input_params)

    tamm.initialize(argv, False)

    try:
        dlpno_test(params)
    finally:
        tamm.finalize(True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
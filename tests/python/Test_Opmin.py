#!/usr/bin/env python3

import sys
import pytamm as tamm


def L(lt):
    return tamm.as_op(lt)


def add_terms(terms):
    if not terms:
        raise ValueError("add_terms requires at least one term")

    out = terms[0]
    for term in terms[1:]:
        out = out + term
    return out


def cs_ccsd_t1(sch):
    IS = tamm.IndexSpace(
        tamm.range(10),
        {
            "occ": [tamm.range(0, 5)],
            "virt": [tamm.range(5, 10)],
        },
    )

    MO = tamm.TiledIndexSpace(IS)

    i, j, m, n = MO.labels("occ", count=4)
    a, e, f = MO.labels("virt", count=3)

    i0 = tamm.TensorDouble([a, i])
    t1 = tamm.TensorDouble([a, m])
    t2 = tamm.TensorDouble([a, e, m, n])
    F = tamm.TensorDouble([MO, MO])
    V = tamm.TensorDouble([MO, MO, MO, MO])

    symbol_table = tamm.SymbolTable()
    tamm.register_symbols(symbol_table, F=F, V=V, t1=t1, t2=t2, i0=i0)
    tamm.register_symbols(symbol_table, i=i, j=j, m=m, n=n, a=a, e=e, f=f)

    sch.allocate(F, V, t1, t2, i0)
    sch(F(), "=", 1.0)
    sch(V(), "=", 1.0)
    sch(t1(), "=", 1.0)
    sch(t2(), "=", 1.0)
    sch(i0(), "=", 1.0)
    sch.execute()

    singles = add_terms(
        [
            L(F(a, i)),

            -2.0 * L(F(m, e)) * L(t1(a, m)) * L(t1(e, i)),

            L(F(a, e)) * L(t1(e, i)),

            -2.0 * L(V(m, n, e, f)) * L(t2(a, f, m, n)) * L(t1(e, i)),

            -2.0
            * L(V(m, n, e, f))
            * L(t1(a, m))
            * L(t1(f, n))
            * L(t1(e, i)),

            L(V(n, m, e, f)) * L(t2(a, f, m, n)) * L(t1(e, i)),

            L(V(n, m, e, f)) * L(t1(a, m)) * L(t1(f, n)) * L(t1(e, i)),

            -1.0 * L(F(m, i)) * L(t1(a, m)),

            -2.0 * L(V(m, n, e, f)) * L(t2(e, f, i, n)) * L(t1(a, m)),

            -2.0
            * L(V(m, n, e, f))
            * L(t1(e, i))
            * L(t1(f, n))
            * L(t1(a, m)),

            L(V(m, n, f, e)) * L(t2(e, f, i, n)) * L(t1(a, m)),

            L(V(m, n, f, e)) * L(t1(e, i)) * L(t1(f, n)) * L(t1(a, m)),

            2.0 * L(F(m, e)) * L(t2(e, a, m, i)),

            -1.0 * L(F(m, e)) * L(t2(e, a, i, m)),

            L(F(m, e)) * L(t1(e, i)) * L(t1(a, m)),

            4.0 * L(V(m, n, e, f)) * L(t1(f, n)) * L(t2(e, a, m, i)),

            -2.0 * L(V(m, n, e, f)) * L(t1(f, n)) * L(t2(e, a, i, m)),

            2.0
            * L(V(m, n, e, f))
            * L(t1(f, n))
            * L(t1(e, i))
            * L(t1(a, m)),

            -2.0 * L(V(m, n, f, e)) * L(t1(f, n)) * L(t2(e, a, m, i)),

            L(V(m, n, f, e)) * L(t1(f, n)) * L(t2(e, a, i, m)),

            -1.0
            * L(V(m, n, f, e))
            * L(t1(f, n))
            * L(t1(e, i))
            * L(t1(a, m)),

            2.0 * L(V(m, a, e, i)) * L(t1(e, m)),

            -1.0 * L(V(m, a, i, e)) * L(t1(e, m)),

            2.0 * L(V(m, a, e, f)) * L(t2(e, f, m, i)),

            2.0 * L(V(m, a, e, f)) * L(t1(e, m)) * L(t1(f, i)),

            -1.0 * L(V(m, a, f, e)) * L(t2(e, f, m, i)),

            -1.0 * L(V(m, a, f, e)) * L(t1(e, m)) * L(t1(f, i)),

            -2.0 * L(V(m, n, e, i)) * L(t2(e, a, m, n)),

            -2.0 * L(V(m, n, e, i)) * L(t1(e, m)) * L(t1(a, n)),

            L(V(n, m, e, i)) * L(t2(e, a, m, n)),

            L(V(n, m, e, i)) * L(t1(e, m)) * L(t1(a, n)),
        ]
    )

    i0(a, i).update(singles)

    op_exec = tamm.OpExecutor(sch, symbol_table)

    op_exec.opmin_execute(i0)
    tamm.print_tensor_all(i0)

    sch(i0(), "=", 1.0)
    sch.execute()

    i0(a, i).update(singles)

    op_exec.execute(i0)
    tamm.print_tensor_all(i0)


def dlpno_test(sch):
    IS = tamm.IndexSpace(
        tamm.range(13),
        {
            "occ": [tamm.range(0, 4)],
            "virt": [tamm.range(4, 13)],
        },
    )

    AO = tamm.TiledIndexSpace(tamm.IndexSpace(tamm.range(13)))
    AO_DF = tamm.TiledIndexSpace(tamm.IndexSpace(tamm.range(375)))
    MO = tamm.TiledIndexSpace(IS)

    PAO = AO("all")
    PNO = MO("virt")
    LMOP = tamm.TiledIndexSpace(tamm.IndexSpace(tamm.range(16)))

    dTEvv = tamm.TensorDouble([PAO, PAO, AO_DF])
    dTEov_00 = tamm.TensorDouble([LMOP, PAO, AO_DF])
    dT1 = tamm.TensorDouble([PNO, LMOP])
    dT2 = tamm.TensorDouble([PNO, PNO, LMOP])
    dT2_out = tamm.TensorDouble([PNO, PNO, LMOP])
    d = tamm.TensorDouble([LMOP, PAO, PNO])
    Siikl = tamm.TensorDouble([LMOP, LMOP, PNO, PNO])

    symbol_table = tamm.SymbolTable()
    tamm.register_symbols(
        symbol_table,
        dTEvv=dTEvv,
        dTEov_00=dTEov_00,
        dT1=dT1,
        dT2=dT2,
        dT2_out=dT2_out,
        d=d,
        Siikl=Siikl,
    )

    sch.allocate(dTEvv, dTEov_00, dT1, dT2, dT2_out, d, Siikl)

    sch(dTEvv(), "=", 1.0)
    sch(dTEov_00(), "=", 1.0)
    sch(dT1(), "=", 1.0)
    sch(dT2(), "=", 1.0)
    sch(dT2_out(), "=", 0.0)
    sch(d(), "=", 1.0)
    sch(Siikl(), "=", 1.0)
    sch.execute()

    dlpno_doubles_12 = (
        -1.0
        * L(dTEvv("a_mu", "e_mu", "K"))
        * L(dTEov_00("mm", "f_mu", "K"))
        * L(dT1("b_mm", "mm"))
        * L(dT2("e_ij", "f_ij", "ij"))
        * L(d("ij", "f_mu", "f_ij"))
        * L(d("ij", "e_mu", "e_ij"))
        * L(d("ij", "a_mu", "a_ij"))
        * L(Siikl("mm", "ij", "b_mm", "b_ij"))
    )

    op_cost = tamm.OpCostCalculator(symbol_table)

    lhs_lt = dT2_out("a_ij", "b_ij", "ij")
    lhs_ltop = tamm.LTOp(lhs_lt)

    print("Print original binarized op")
    op_cost.print_op_binarized(lhs_ltop, dlpno_doubles_12.clone())

    original_op_cost = op_cost.get_op_cost(
        dlpno_doubles_12.clone(),
        lhs_lt,
    )
    print(f"Original op cost: {original_op_cost}")

    opmin = tamm.OpMin(symbol_table)

    optimized_dlpno_doubles_12 = opmin.optimize_all(
        lhs_ltop,
        dlpno_doubles_12,
    )

    print("Print opmined binarized op")
    op_cost.print_op_binarized(
        tamm.LTOp(dT2_out("a_ij", "b_ij", "ij")),
        optimized_dlpno_doubles_12,
    )

    opmined_op_cost = op_cost.get_op_cost(
        optimized_dlpno_doubles_12.clone(),
        dT2_out("a_ij", "b_ij", "ij"),
    )
    print(f"Opmined op cost: {opmined_op_cost}")


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

        # C++ main has this commented out:
        # cs_ccsd_t1(sch)

        dlpno_test(sch)

    finally:
        tamm.finalize(True)


if __name__ == "__main__":
    main()
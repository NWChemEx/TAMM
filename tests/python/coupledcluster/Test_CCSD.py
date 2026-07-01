#!/usr/bin/env python3

import datetime
import math
import sys
import time

import pytamm as tamm


T = float


# -----------------------------------------------------------------------------
# Small Python replacement for the C++ CCSE_Tensors helper used by the test.
# -----------------------------------------------------------------------------
class CCSE_Tensors:
    def __init__(self, mo=None, spaces=None, name="", spin_blocks=()):
        self.mo = mo
        self.spaces = list(spaces or [])
        self.name = name
        self.blocks = {}

        for spin in spin_blocks:
            self.blocks[spin] = tamm.TensorDouble(self.spaces)

    def __call__(self, spin):
        return self.blocks[spin]

    def tensors(self):
        return list(self.blocks.values())

    @staticmethod
    def allocate_list(sch, *objs):
        tensors = []
        for obj in objs:
            tensors.extend(obj.tensors())
        if tensors:
            sch.allocate(*tensors)

    @staticmethod
    def deallocate_list(sch, *objs):
        tensors = []
        for obj in objs:
            tensors.extend(obj.tensors())
        if tensors:
            sch.deallocate(*tensors)


def labels(tis, label_id="all", count=1):
    return tis.labels(label_id, None, count)


def sop(sch, lhs, op, rhs, opstr=None):
    if opstr is None:
        return sch(lhs, op, rhs)
    return sch(lhs, op, rhs, opstr=opstr)


# -----------------------------------------------------------------------------
# Globals matching the C++ test globals.
# -----------------------------------------------------------------------------
_a021 = None
a22_abab = None

o_alpha = None
v_alpha = None
o_beta = None
v_beta = None

_a01V = None
_a02V = None
_a007V = None

_a01 = None
_a02 = None
_a03 = None
_a04 = None
_a05 = None
_a06 = None
_a001 = None
_a004 = None
_a006 = None
_a008 = None
_a009 = None
_a017 = None
_a019 = None
_a020 = None

i0_temp = None
t2_aaaa_temp = None


# -----------------------------------------------------------------------------
# setupTensors_cs
# -----------------------------------------------------------------------------
def setup_tensors_cs(ec, MO, d_f1):
    global o_alpha, v_alpha, o_beta, v_beta

    O = MO("occ")
    V = MO("virt")

    otiles = O.num_tiles()
    vtiles = V.num_tiles()

    oatiles = MO("occ_alpha").num_tiles()
    vatiles = MO("virt_alpha").num_tiles()

    o_alpha = tamm.TiledIndexSpace(MO("occ"), tamm.range(oatiles))
    v_alpha = tamm.TiledIndexSpace(MO("virt"), tamm.range(vatiles))
    o_beta = tamm.TiledIndexSpace(MO("occ"), tamm.range(oatiles, otiles))
    v_beta = tamm.TiledIndexSpace(MO("virt"), tamm.range(vatiles, vtiles))

    p_evl_sorted = tamm.diagonal(d_f1)

    d_r1 = tamm.TensorDouble([v_alpha, o_alpha], [1, 1])
    d_r2 = tamm.TensorDouble([v_alpha, v_beta, o_alpha, o_beta], [2, 2])
    tamm.TensorDouble.allocate(ec, d_r1, d_r2)

    d_t1 = tamm.TensorDouble([v_alpha, o_alpha], [1, 1])
    d_t2 = tamm.TensorDouble([v_alpha, v_beta, o_alpha, o_beta], [2, 2])
    tamm.TensorDouble.allocate(ec, d_t1, d_t2)

    return p_evl_sorted, d_t1, d_t2, d_r1, d_r2


# -----------------------------------------------------------------------------
# ccsd_e_cs
# -----------------------------------------------------------------------------
def ccsd_e_cs(sch, MO, CI, de, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se):
    global _a01V, _a02, _a03, t2_aaaa_temp
    global o_alpha, v_alpha, o_beta, v_beta

    (cind,) = labels(CI, "all", 1)

    p1_va, p2_va = labels(v_alpha, "all", 2)
    (p1_vb,) = labels(v_beta, "all", 1)

    h1_oa, h2_oa = labels(o_alpha, "all", 2)
    (h1_ob,) = labels(o_beta, "all", 1)

    f1_ov = f1_se[1]
    chol3d_ov = chol3d_se[1]

    sop(sch, t2_aaaa_temp(), "=", 0)

    sch.exact_copy(
        t2_aaaa(p1_va, p2_va, h1_oa, h2_oa),
        t2_abab(p1_va, p2_va, h1_oa, h2_oa),
    )

    sop(sch, t2_aaaa_temp(), "=", t2_aaaa())

    sop(
        sch,
        t2_aaaa(p1_va, p2_va, h1_oa, h2_oa),
        "+=",
        -1.0 * t2_aaaa_temp(p2_va, p1_va, h1_oa, h2_oa),
    )

    sop(
        sch,
        t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa),
        "+=",
        1.0 * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa),
    )

    sop(
        sch,
        _a01V(cind),
        "=",
        t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    )

    sop(
        sch,
        _a02("aa")(h1_oa, h2_oa, cind),
        "=",
        t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind),
    )

    sop(
        sch,
        _a03("aa")(h2_oa, p2_va, cind),
        "=",
        t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa)
        * chol3d_ov("aa")(h1_oa, p1_va, cind),
    )

    sop(sch, de(), "=", 2.0 * _a01V() * _a01V())

    sop(
        sch,
        de(),
        "+=",
        -1.0
        * _a02("aa")(h1_oa, h2_oa, cind)
        * _a02("aa")(h2_oa, h1_oa, cind),
    )

    sop(
        sch,
        de(),
        "+=",
        1.0 * _a03("aa")(h1_oa, p1_va, cind) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    )

    sop(
        sch,
        de(),
        "+=",
        2.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    )


# -----------------------------------------------------------------------------
# ccsd_t1_cs
# -----------------------------------------------------------------------------
def ccsd_t1_cs(sch, MO, CI, i0_aa, t1_aa, t2_abab, f1_se, chol3d_se):
    global _a01, _a02V, _a04, _a05, _a06, t2_aaaa_temp
    global o_alpha, v_alpha, o_beta, v_beta

    (cind,) = labels(CI, "all", 1)

    p1_va, p2_va = labels(v_alpha, "all", 2)
    (p1_vb,) = labels(v_beta, "all", 1)

    h1_oa, h2_oa = labels(o_alpha, "all", 2)
    (h1_ob,) = labels(o_beta, "all", 1)

    f1_oo = f1_se[0]
    f1_ov = f1_se[1]
    f1_vv = f1_se[2]

    chol3d_oo = chol3d_se[0]
    chol3d_ov = chol3d_se[1]
    chol3d_vv = chol3d_se[2]

    sop(
        sch,
        i0_aa(p2_va, h1_oa),
        "=",
        1.0 * f1_ov("aa")(h1_oa, p2_va),
    )

    sop(
        sch,
        _a01("aa")(h2_oa, h1_oa, cind),
        "=",
        1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind),
    )

    sop(
        sch,
        _a02V(cind),
        "=",
        2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    )

    sop(
        sch,
        _a05("aa")(h2_oa, p1_va),
        "=",
        -1.0
        * chol3d_ov("aa")(h1_oa, p1_va, cind)
        * _a01("aa")(h2_oa, h1_oa, cind),
    )

    sop(
        sch,
        _a05("aa")(h2_oa, p1_va),
        "+=",
        1.0 * f1_ov("aa")(h2_oa, p1_va),
    )

    sop(
        sch,
        _a06("aa")(p1_va, h1_oa, cind),
        "=",
        -1.0
        * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa)
        * chol3d_ov("aa")(h2_oa, p2_va, cind),
    )

    sop(
        sch,
        _a04("aa")(h2_oa, h1_oa),
        "=",
        -1.0 * f1_oo("aa")(h2_oa, h1_oa),
    )

    sop(
        sch,
        _a04("aa")(h2_oa, h1_oa),
        "+=",
        1.0
        * chol3d_ov("aa")(h2_oa, p1_va, cind)
        * _a06("aa")(p1_va, h1_oa, cind),
    )

    sop(
        sch,
        _a04("aa")(h2_oa, h1_oa),
        "+=",
        -1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    )

    sop(
        sch,
        i0_aa(p2_va, h1_oa),
        "+=",
        1.0 * t1_aa(p2_va, h2_oa) * _a04("aa")(h2_oa, h1_oa),
    )

    sop(
        sch,
        i0_aa(p1_va, h2_oa),
        "+=",
        1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a02V(cind),
    )

    sop(
        sch,
        i0_aa(p1_va, h2_oa),
        "+=",
        1.0
        * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa)
        * _a05("aa")(h1_oa, p2_va),
    )

    sop(
        sch,
        i0_aa(p2_va, h1_oa),
        "+=",
        -1.0
        * chol3d_vv("aa")(p2_va, p1_va, cind)
        * _a06("aa")(p1_va, h1_oa, cind),
    )

    sop(
        sch,
        _a06("aa")(p2_va, h2_oa, cind),
        "+=",
        -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv("aa")(p2_va, p1_va, cind),
    )

    sop(
        sch,
        i0_aa(p1_va, h2_oa),
        "+=",
        -1.0 * _a06("aa")(p1_va, h2_oa, cind) * _a02V(cind),
    )

    sop(
        sch,
        _a06("aa")(p2_va, h1_oa, cind),
        "+=",
        -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind),
    )

    sop(
        sch,
        _a06("aa")(p2_va, h1_oa, cind),
        "+=",
        1.0 * t1_aa(p2_va, h2_oa) * _a01("aa")(h2_oa, h1_oa, cind),
    )

    sop(
        sch,
        _a01("aa")(h2_oa, h1_oa, cind),
        "+=",
        1.0 * chol3d_oo("aa")(h2_oa, h1_oa, cind),
    )

    sop(
        sch,
        i0_aa(p2_va, h1_oa),
        "+=",
        1.0
        * _a01("aa")(h2_oa, h1_oa, cind)
        * _a06("aa")(p2_va, h2_oa, cind),
    )

    sop(
        sch,
        i0_aa(p2_va, h1_oa),
        "+=",
        1.0 * t1_aa(p1_va, h1_oa) * f1_vv("aa")(p2_va, p1_va),
    )


# -----------------------------------------------------------------------------
# ccsd_t2_cs
# -----------------------------------------------------------------------------
def ccsd_t2_cs(sch, MO, CI, i0_abab, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se):
    global a22_abab
    global _a001, _a004, _a006, _a007V, _a008, _a009
    global _a017, _a019, _a020, _a021
    global i0_temp
    global o_alpha, v_alpha, o_beta, v_beta

    (cind,) = labels(CI, "all", 1)

    p1_va, p2_va, p3_va = labels(v_alpha, "all", 3)
    p1_vb, p2_vb = labels(v_beta, "all", 2)

    h1_oa, h2_oa, h3_oa = labels(o_alpha, "all", 3)
    h1_ob, h2_ob = labels(o_beta, "all", 2)

    f1_oo = f1_se[0]
    f1_ov = f1_se[1]
    f1_vv = f1_se[2]

    chol3d_oo = chol3d_se[0]
    chol3d_ov = chol3d_se[1]
    chol3d_vv = chol3d_se[2]

    # The binding layer does not expose the custom AddBuf/kernel-backed callback
    # used in the C++ test, so materialize this intermediate as a normal tensor.
    a22_abab = tamm.TensorDouble([v_alpha, v_beta, v_alpha, v_beta])
    sch.allocate(a22_abab)

    sop(
        sch,
        _a017("aa")(p1_va, h2_oa, cind),
        "=",
        -1.0
        * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa)
        * chol3d_ov("aa")(h1_oa, p2_va, cind),
    )

    sop(
        sch,
        _a006("aa")(h2_oa, h1_oa),
        "=",
        -1.0
        * chol3d_ov("aa")(h2_oa, p2_va, cind)
        * _a017("aa")(p2_va, h1_oa, cind),
    )

    sop(
        sch,
        _a007V(cind),
        "=",
        2.0 * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa),
    )

    sop(
        sch,
        _a009("aa")(h1_oa, h2_oa, cind),
        "=",
        1.0 * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa),
    )

    sop(
        sch,
        _a021("aa")(p2_va, p1_va, cind),
        "=",
        -0.5 * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa),
    )

    sop(
        sch,
        _a021("aa")(p2_va, p1_va, cind),
        "+=",
        0.5 * chol3d_vv("aa")(p2_va, p1_va, cind),
    )

    sop(
        sch,
        _a017("aa")(p1_va, h2_oa, cind),
        "+=",
        -2.0 * t1_aa(p2_va, h2_oa) * _a021("aa")(p1_va, p2_va, cind),
    )

    sop(
        sch,
        _a008("aa")(h2_oa, h1_oa, cind),
        "=",
        1.0 * _a009("aa")(h2_oa, h1_oa, cind),
    )

    sop(
        sch,
        _a009("aa")(h2_oa, h1_oa, cind),
        "+=",
        1.0 * chol3d_oo("aa")(h2_oa, h1_oa, cind),
    )

    sch.exact_copy(
        _a009("bb")(h2_ob, h1_ob, cind),
        _a009("aa")(h2_ob, h1_ob, cind),
    )

    sch.exact_copy(
        _a021("bb")(p2_vb, p1_vb, cind),
        _a021("aa")(p2_vb, p1_vb, cind),
    )

    sop(
        sch,
        _a001("aa")(p1_va, p2_va),
        "=",
        -2.0 * _a021("aa")(p1_va, p2_va, cind) * _a007V(cind),
    )

    sop(
        sch,
        _a001("aa")(p1_va, p2_va),
        "+=",
        -1.0
        * _a017("aa")(p1_va, h2_oa, cind)
        * chol3d_ov("aa")(h2_oa, p2_va, cind),
    )

    sop(
        sch,
        _a006("aa")(h2_oa, h1_oa),
        "+=",
        1.0 * _a009("aa")(h2_oa, h1_oa, cind) * _a007V(cind),
    )

    sop(
        sch,
        _a006("aa")(h3_oa, h1_oa),
        "+=",
        -1.0
        * _a009("aa")(h2_oa, h1_oa, cind)
        * _a008("aa")(h3_oa, h2_oa, cind),
    )

    sop(
        sch,
        _a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob),
        "=",
        0.25
        * _a009("aa")(h2_oa, h1_oa, cind)
        * _a009("bb")(h1_ob, h2_ob, cind),
    )

    sop(
        sch,
        _a020("aaaa")(p2_va, h2_oa, p1_va, h1_oa),
        "=",
        -2.0
        * _a009("aa")(h2_oa, h1_oa, cind)
        * _a021("aa")(p2_va, p1_va, cind),
    )

    sch.exact_copy(
        _a020("baba")(p2_vb, h2_oa, p1_vb, h1_oa),
        _a020("aaaa")(p2_vb, h2_oa, p1_vb, h1_oa),
    )

    sop(
        sch,
        _a020("aaaa")(p1_va, h3_oa, p3_va, h2_oa),
        "+=",
        0.5
        * _a004("aaaa")(p2_va, p3_va, h3_oa, h1_oa)
        * t2_aaaa(p1_va, p2_va, h1_oa, h2_oa),
    )

    sop(
        sch,
        _a020("baab")(p1_vb, h2_oa, p1_va, h2_ob),
        "=",
        -0.5
        * _a004("aaaa")(p2_va, p1_va, h2_oa, h1_oa)
        * t2_abab(p2_va, p1_vb, h1_oa, h2_ob),
    )

    sop(
        sch,
        _a020("baba")(p1_vb, h1_oa, p2_vb, h2_oa),
        "+=",
        0.5
        * _a004("abab")(p1_va, p2_vb, h1_oa, h1_ob)
        * t2_abab(p1_va, p1_vb, h2_oa, h1_ob),
    )

    sop(
        sch,
        _a017("aa")(p1_va, h2_oa, cind),
        "+=",
        1.0 * t1_aa(p1_va, h1_oa) * chol3d_oo("aa")(h1_oa, h2_oa, cind),
    )

    sop(
        sch,
        _a017("aa")(p1_va, h2_oa, cind),
        "+=",
        -1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind),
    )

    sop(
        sch,
        _a001("aa")(p2_va, p1_va),
        "+=",
        -1.0 * f1_vv("aa")(p2_va, p1_va),
    )

    sop(
        sch,
        _a001("aa")(p2_va, p1_va),
        "+=",
        1.0 * t1_aa(p2_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    )

    sop(
        sch,
        _a006("aa")(h2_oa, h1_oa),
        "+=",
        1.0 * f1_oo("aa")(h2_oa, h1_oa),
    )

    sop(
        sch,
        _a006("aa")(h2_oa, h1_oa),
        "+=",
        1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    )

    sch.exact_copy(
        _a017("bb")(p1_vb, h1_ob, cind),
        _a017("aa")(p1_vb, h1_ob, cind),
    )

    sch.exact_copy(
        _a006("bb")(h1_ob, h2_ob),
        _a006("aa")(h1_ob, h2_ob),
    )

    sch.exact_copy(
        _a001("bb")(p1_vb, p2_vb),
        _a001("aa")(p1_vb, p2_vb),
    )

    sch.exact_copy(
        _a021("bb")(p1_vb, p2_vb, cind),
        _a021("aa")(p1_vb, p2_vb, cind),
    )

    sch.exact_copy(
        _a020("bbbb")(p1_vb, h1_ob, p2_vb, h2_ob),
        _a020("aaaa")(p1_vb, h1_ob, p2_vb, h2_ob),
    )

    sop(
        sch,
        i0_abab(p1_va, p2_vb, h2_oa, h1_ob),
        "=",
        1.0
        * _a020("bbbb")(p2_vb, h2_ob, p1_vb, h1_ob)
        * t2_abab(p1_va, p1_vb, h2_oa, h2_ob),
    )

    sop(
        sch,
        i0_abab(p2_va, p1_vb, h2_oa, h1_ob),
        "+=",
        1.0
        * _a020("baab")(p1_vb, h1_oa, p1_va, h1_ob)
        * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa),
    )

    sop(
        sch,
        i0_abab(p1_va, p1_vb, h2_oa, h1_ob),
        "+=",
        1.0
        * _a020("baba")(p1_vb, h1_oa, p2_vb, h2_oa)
        * t2_abab(p1_va, p2_vb, h1_oa, h1_ob),
    )

    sch.exact_copy(
        i0_temp(p1_vb, p1_va, h2_ob, h1_oa),
        i0_abab(p1_vb, p1_va, h2_ob, h1_oa),
    )

    sop(
        sch,
        i0_abab(p1_va, p1_vb, h2_oa, h1_ob),
        "+=",
        1.0 * i0_temp(p1_vb, p1_va, h1_ob, h2_oa),
    )

    sop(
        sch,
        i0_abab(p1_va, p1_vb, h1_oa, h2_ob),
        "+=",
        1.0
        * _a017("aa")(p1_va, h1_oa, cind)
        * _a017("bb")(p1_vb, h2_ob, cind),
    )

    # Materialized replacement for the callback-backed a22_abab.
    sop(
        sch,
        a22_abab(p1_va, p2_vb, p2_va, p1_vb),
        "=",
        1.0
        * _a021("aa")(p1_va, p2_va, cind)
        * _a021("bb")(p2_vb, p1_vb, cind),
    )

    sop(
        sch,
        i0_abab(p1_va, p2_vb, h1_oa, h2_ob),
        "+=",
        4.0
        * a22_abab(p1_va, p2_vb, p2_va, p1_vb)
        * t2_abab(p2_va, p1_vb, h1_oa, h2_ob),
    )

    sop(
        sch,
        _a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob),
        "+=",
        0.25
        * _a004("abab")(p1_va, p2_vb, h2_oa, h1_ob)
        * t2_abab(p1_va, p2_vb, h1_oa, h2_ob),
    )

    sop(
        sch,
        i0_abab(p1_va, p1_vb, h1_oa, h2_ob),
        "+=",
        4.0
        * _a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob)
        * t2_abab(p1_va, p1_vb, h2_oa, h1_ob),
    )

    sop(
        sch,
        i0_abab(p1_va, p1_vb, h1_oa, h2_ob),
        "+=",
        -1.0 * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001("bb")(p1_vb, p2_vb),
    )

    sop(
        sch,
        i0_abab(p1_va, p1_vb, h1_oa, h2_ob),
        "+=",
        -1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001("aa")(p1_va, p2_va),
    )

    sop(
        sch,
        i0_abab(p1_va, p1_vb, h2_oa, h1_ob),
        "+=",
        -1.0 * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006("aa")(h1_oa, h2_oa),
    )

    sop(
        sch,
        i0_abab(p1_va, p1_vb, h1_oa, h2_ob),
        "+=",
        -1.0 * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006("bb")(h1_ob, h2_ob),
    )


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(argv):
    global _a021, a22_abab
    global o_alpha, v_alpha, o_beta, v_beta
    global _a01V, _a02V, _a007V
    global _a01, _a02, _a03, _a04, _a05, _a06
    global _a001, _a004, _a006, _a008, _a009, _a017, _a019, _a020
    global i0_temp, t2_aaaa_temp

    tamm.initialize(argv)

    if len(argv) < 5:
        tamm.finalize()
        raise SystemExit("Please provide occ_alpha, virt_alpha, cholesky-count and tile size")

    n_occ_alpha = int(argv[1])
    n_vir_alpha = int(argv[2])
    chol_count = int(argv[3])
    tile_size = int(argv[4])

    nbf = n_occ_alpha + n_vir_alpha

    pg = tamm.ProcGroup.create_world_coll()
    ec = tamm.ExecutionContext(pg, tamm.DistributionKind.nw, tamm.MemoryManagerKind.ga)
    exhw = ec.exhw()
    sch = tamm.Scheduler(ec)

    profile = True

    if ec.print():
        print(tamm.tamm_git_info())

        print()
        print("date:", datetime.datetime.now().strftime("%c"))
        print(f"nnodes: {ec.nnodes()}, ", end="")
        print(f"nproc_per_node: {ec.ppn()}, ", end="")
        print(f"nproc_total: {ec.nnodes() * ec.ppn()}, ")

        print()
        ec.print_mem_info()
        print()

        print(
            f"basis functions: {nbf}, occ_alpha: {n_occ_alpha}, "
            f"virt_alpha: {n_vir_alpha}, chol-count: {chol_count}, "
            f"tilesize: {tile_size}"
        )

    n_occ_beta = n_occ_alpha
    tce_tile = tile_size
    nmo = 2 * nbf
    n_vir_beta = n_vir_alpha
    nocc = 2 * n_occ_alpha
    total_orbitals = nmo

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

    mo_tiles = []

    est_nt = int(math.ceil(1.0 * n_occ_alpha / tce_tile))
    for x in range(est_nt):
        mo_tiles.append(n_occ_alpha // est_nt + (1 if x < (n_occ_alpha % est_nt) else 0))

    est_nt = int(math.ceil(1.0 * n_occ_beta / tce_tile))
    for x in range(est_nt):
        mo_tiles.append(n_occ_beta // est_nt + (1 if x < (n_occ_beta % est_nt) else 0))

    est_nt = int(math.ceil(1.0 * n_vir_alpha / tce_tile))
    for x in range(est_nt):
        mo_tiles.append(n_vir_alpha // est_nt + (1 if x < (n_vir_alpha % est_nt) else 0))

    est_nt = int(math.ceil(1.0 * n_vir_beta / tce_tile))
    for x in range(est_nt):
        mo_tiles.append(n_vir_beta // est_nt + (1 if x < (n_vir_beta % est_nt) else 0))

    MO = tamm.TiledIndexSpace(MO_IS, mo_tiles)

    N = MO("all")

    d_f1 = tamm.TensorDouble([N, N], [1, 1])
    tamm.TensorDouble.allocate(ec, d_f1)
    tamm.random_ip(d_f1)

    p_evl_sorted, t1_aa, t2_abab, r1_aa, r2_abab = setup_tensors_cs(ec, MO, d_f1)

    chol_is = tamm.IndexSpace(tamm.range(0, chol_count))
    CI = tamm.TiledIndexSpace(chol_is, 1000)

    O = MO("occ")
    V = MO("virt")

    (cind,) = labels(CI, "all", 1)

    otiles = O.num_tiles()
    vtiles = V.num_tiles()

    oatiles = MO("occ_alpha").num_tiles()
    vatiles = MO("virt_alpha").num_tiles()

    o_alpha = tamm.TiledIndexSpace(MO("occ"), tamm.range(oatiles))
    v_alpha = tamm.TiledIndexSpace(MO("virt"), tamm.range(vatiles))
    o_beta = tamm.TiledIndexSpace(MO("occ"), tamm.range(oatiles, otiles))
    v_beta = tamm.TiledIndexSpace(MO("virt"), tamm.range(vatiles, vtiles))

    p1_va, p2_va = labels(v_alpha, "all", 2)
    p1_vb, p2_vb = labels(v_beta, "all", 2)

    h3_oa, h4_oa = labels(o_alpha, "all", 2)
    h3_ob, h4_ob = labels(o_beta, "all", 2)

    d_e = tamm.TensorDouble()

    t2_aaaa = tamm.TensorDouble([v_alpha, v_alpha, o_alpha, o_alpha], [2, 2])

    f1_oo = CCSE_Tensors(MO, [O, O], "f1_oo", ["aa"])
    f1_ov = CCSE_Tensors(MO, [O, V], "f1_ov", ["aa"])
    f1_vv = CCSE_Tensors(MO, [V, V], "f1_vv", ["aa"])

    chol3d_oo = CCSE_Tensors(MO, [O, O, CI], "chol3d_oo", ["aa"])
    chol3d_ov = CCSE_Tensors(MO, [O, V, CI], "chol3d_ov", ["aa"])
    chol3d_vv = CCSE_Tensors(MO, [V, V, CI], "chol3d_vv", ["aa"])

    f1_se = [f1_oo, f1_ov, f1_vv]
    chol3d_se = [chol3d_oo, chol3d_ov, chol3d_vv]

    _a01V = tamm.TensorDouble([CI])
    _a02 = CCSE_Tensors(MO, [O, O, CI], "_a02", ["aa"])
    _a03 = CCSE_Tensors(MO, [O, V, CI], "_a03", ["aa"])
    _a004 = CCSE_Tensors(MO, [V, V, O, O], "_a004", ["aaaa", "abab"])

    t2_aaaa_temp = tamm.TensorDouble([v_alpha, v_alpha, o_alpha, o_alpha])
    i0_temp = tamm.TensorDouble([v_beta, v_alpha, o_beta, o_alpha])

    _a02V = tamm.TensorDouble([CI])
    _a01 = CCSE_Tensors(MO, [O, O, CI], "_a01", ["aa"])
    _a04 = CCSE_Tensors(MO, [O, O], "_a04", ["aa"])
    _a05 = CCSE_Tensors(MO, [O, V], "_a05", ["aa"])
    _a06 = CCSE_Tensors(MO, [V, O, CI], "_a06", ["aa"])

    _a007V = tamm.TensorDouble([CI])
    _a001 = CCSE_Tensors(MO, [V, V], "_a001", ["aa", "bb"])
    _a006 = CCSE_Tensors(MO, [O, O], "_a006", ["aa", "bb"])
    _a008 = CCSE_Tensors(MO, [O, O, CI], "_a008", ["aa"])
    _a009 = CCSE_Tensors(MO, [O, O, CI], "_a009", ["aa", "bb"])
    _a017 = CCSE_Tensors(MO, [V, O, CI], "_a017", ["aa", "bb"])
    _a021 = CCSE_Tensors(MO, [V, V, CI], "_a021", ["aa", "bb"])
    _a019 = CCSE_Tensors(MO, [O, O, O, O], "_a019", ["abab"])
    _a020 = CCSE_Tensors(MO, [V, O, V, O], "_a020", ["aaaa", "baba", "baab", "bbbb"])

    sch.allocate(t2_aaaa)

    sch.allocate(d_e, i0_temp, t2_aaaa_temp, _a01V)

    CCSE_Tensors.allocate_list(
        sch,
        f1_oo,
        f1_ov,
        f1_vv,
        chol3d_oo,
        chol3d_ov,
        chol3d_vv,
    )

    CCSE_Tensors.allocate_list(sch, _a02, _a03)

    sch.allocate(_a02V, _a007V)

    CCSE_Tensors.allocate_list(
        sch,
        _a004,
        _a01,
        _a04,
        _a05,
        _a06,
        _a001,
        _a006,
        _a008,
        _a009,
        _a017,
        _a019,
        _a020,
        _a021,
    )

    sch.execute()

    tamm.random_ip(f1_oo("aa"))
    tamm.random_ip(f1_ov("aa"))
    tamm.random_ip(f1_vv("aa"))
    tamm.random_ip(chol3d_oo("aa"))
    tamm.random_ip(chol3d_ov("aa"))
    tamm.random_ip(chol3d_vv("aa"))

    sop(
        sch,
        _a004("aaaa")(p1_va, p2_va, h4_oa, h3_oa),
        "=",
        1.0
        * chol3d_ov("aa")(h4_oa, p1_va, cind)
        * chol3d_ov("aa")(h3_oa, p2_va, cind),
    )

    sch.exact_copy(
        _a004("abab")(p1_va, p1_vb, h3_oa, h3_ob),
        _a004("aaaa")(p1_va, p1_vb, h3_oa, h3_ob),
    )

    sch.execute(exhw)

    timer_start = time.perf_counter()

    ccsd_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se)
    ccsd_t1_cs(sch, MO, CI, r1_aa, t1_aa, t2_abab, f1_se, chol3d_se)
    ccsd_t2_cs(sch, MO, CI, r2_abab, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se)

    sch.execute(exhw, profile)

    timer_end = time.perf_counter()
    iter_time = timer_end - timer_start

    if ec.print():
        print(f"Time taken for closed-shell CD-CCSD: {iter_time}")

    if profile and ec.print():
        profile_csv = (
            f"ccsd_profile_{nbf}bf_"
            f"{n_occ_alpha}oa_"
            f"{n_vir_alpha}va_"
            f"{chol_count}cv_"
            f"{tile_size}TS.csv"
        )

        with open(profile_csv, "w", encoding="utf-8") as pds:
            pds.write(ec.get_profile_header())
            pds.write("\n")
            pds.write(ec.get_profile_data())
            pds.write("\n")

    if False:
        ec_dense = tamm.ExecutionContext(
            ec.pg(),
            tamm.DistributionKind.dense,
            tamm.MemoryManagerKind.ga,
        )
        c3dvv_dense = tamm.to_dense_tensor(ec_dense, chol3d_vv("aa"))
        tamm.print_tensor(c3dvv_dense, "c3d_vv_aa_dense")

    sch.deallocate(_a02V, _a007V)

    CCSE_Tensors.deallocate_list(
        sch,
        _a004,
        _a01,
        _a04,
        _a05,
        _a06,
        _a001,
        _a006,
        _a008,
        _a009,
        _a017,
        _a019,
        _a020,
        _a021,
    )

    if a22_abab is not None:
        sch.deallocate(a22_abab)

    sch.deallocate(d_e, i0_temp, t2_aaaa_temp, _a01V)

    CCSE_Tensors.deallocate_list(sch, _a02, _a03)

    CCSE_Tensors.deallocate_list(
        sch,
        f1_oo,
        f1_ov,
        f1_vv,
        chol3d_oo,
        chol3d_ov,
        chol3d_vv,
    )

    sch.execute()

    sch.deallocate(t1_aa, t2_abab, r1_aa, r2_abab, d_f1, t2_aaaa).execute()

    tamm.finalize()


if __name__ == "__main__":
    main(sys.argv)
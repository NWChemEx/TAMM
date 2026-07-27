#!/usr/bin/env python3

import datetime
import math
import re
import sys
import time

import pytamm as tamm


# -----------------------------------------------------------------------------
# Globals matching the C++ file
# -----------------------------------------------------------------------------

CCEType = float

o_alpha = None
v_alpha = None
o_beta = None
v_beta = None

_a01V = None
_a02V = None
_a007V = None

_a01_sp = None
_a02_sp = None
_a03_sp = None
_a04_sp = None
_a05_sp = None
_a06_sp = None
_a001_sp = None
_a004_sp = None
_a006_sp = None
_a008_sp = None
_a009_sp = None
_a017_sp = None
_a019_sp = None
_a020_sp = None
_a021_sp = None
_a022_sp = None

i0_temp = None
t2_aaaa_temp = None

index_to_sub_string = {
    "I": "occ_alpha",
    "J": "occ_alpha",
    "K": "occ_alpha",
    "L": "occ_alpha",
    "i": "occ_beta",
    "j": "occ_beta",
    "k": "occ_beta",
    "l": "occ_beta",
    "A": "virt_alpha",
    "B": "virt_alpha",
    "C": "virt_alpha",
    "D": "virt_alpha",
    "a": "virt_beta",
    "b": "virt_beta",
    "c": "virt_beta",
    "d": "virt_beta",
    "X": "all",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def atoi_cpp(value):
    s = str(value)
    m = re.match(r"^\s*([+-]?\d+)", s)
    return int(m.group(1)) if m else 0


def exact_copy(sch, lhs, rhs, update=False):
    if not hasattr(sch, "exact_copy"):
        raise RuntimeError(
            "Scheduler.exact_copy is required for this test. "
            "Do not replace exact_copy with normal assignment because this test "
            "copies between slices of the same tensor."
        )
    return sch.exact_copy(lhs, rhs, update)


def generate_spin_check(t_spaces, spin_mask):
    assert len(t_spaces) == len(spin_mask)

    def is_non_zero_spin(blockid):
        upper_total = 0
        lower_total = 0
        other_total = 0

        for idx in range(len(blockid)):
            spin_val = int(t_spaces[idx].spin(blockid[idx]))

            if spin_mask[idx] == tamm.SpinPosition.upper:
                upper_total += spin_val
            elif spin_mask[idx] == tamm.SpinPosition.lower:
                lower_total += spin_val
            else:
                other_total += spin_val

        return upper_total == lower_total

    return is_non_zero_spin


def balanced_tiles(n, tce_tile):
    est_nt = int(math.ceil(1.0 * n / tce_tile))
    out = []
    for x in range(est_nt):
        out.append(n // est_nt + (1 if x < (n % est_nt) else 0))
    return out


def make_mo(n_occ_alpha, n_vir_alpha, tile_size):
    n_occ_beta = n_occ_alpha
    tce_tile = tile_size
    nbf = n_occ_alpha + n_vir_alpha
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
    mo_tiles.extend(balanced_tiles(n_occ_alpha, tce_tile))
    mo_tiles.extend(balanced_tiles(n_occ_beta, tce_tile))
    mo_tiles.extend(balanced_tiles(n_vir_alpha, tce_tile))
    mo_tiles.extend(balanced_tiles(n_vir_beta, tce_tile))

    return tamm.TiledIndexSpace(MO_IS, mo_tiles)


def print_header(ec, nbf, n_occ_alpha, n_vir_alpha, chol_count, tile_size):
    if not ec.print():
        return

    print(tamm.tamm_git_info())

    now = datetime.datetime.now()
    print()
    print("date:", now.strftime("%c"))

    print(f"nnodes: {ec.nnodes()}, ", end="")
    print(f"nproc_per_node: {ec.ppn()}, ", end="")
    print(f"nproc_total: {ec.nnodes() * ec.ppn()}, ", end="")

    if ec.has_gpu():
        if hasattr(ec, "gpn"):
            print(f"ngpus_per_node: {ec.gpn()}, ", end="")
            print(f"ngpus_total: {ec.nnodes() * ec.gpn()}")
        else:
            print("gpu info unavailable")

    print()
    ec.print_mem_info()
    print()

    print(
        f"basis functions: {nbf}, occ_alpha: {n_occ_alpha}, "
        f"virt_alpha: {n_vir_alpha}, chol-count: {chol_count}, "
        f"tilesize: {tile_size}"
    )


# -----------------------------------------------------------------------------
# setupTensors_cs
# -----------------------------------------------------------------------------

def setupTensors_cs(ec, MO, d_f1):
    global o_alpha, v_alpha, o_beta, v_beta

    O = MO("occ")
    V = MO("virt")

    otiles = int(O.num_tiles())
    vtiles = int(V.num_tiles())
    oatiles = int(MO("occ_alpha").num_tiles())
    vatiles = int(MO("virt_alpha").num_tiles())

    o_alpha = tamm.TiledIndexSpace(MO("occ"), tamm.range(oatiles))
    v_alpha = tamm.TiledIndexSpace(MO("virt"), tamm.range(vatiles))
    o_beta = tamm.TiledIndexSpace(MO("occ"), tamm.range(oatiles, otiles))
    v_beta = tamm.TiledIndexSpace(MO("virt"), tamm.range(vatiles, vtiles))

    p_evl_sorted = tamm.diagonal(d_f1)

    non_zero_check_Va_Oa = generate_spin_check(
        [v_alpha, o_alpha],
        [tamm.SpinPosition.lower, tamm.SpinPosition.upper],
    )
    non_zero_check_Va_Vb_Oa_Ob = generate_spin_check(
        [v_alpha, v_beta, o_alpha, o_beta],
        [
            tamm.SpinPosition.lower,
            tamm.SpinPosition.lower,
            tamm.SpinPosition.upper,
            tamm.SpinPosition.upper,
        ],
    )

    block_info_Va_Oa = tamm.TensorInfo([v_alpha, o_alpha], non_zero_check_Va_Oa)
    block_info_Va_Vb_Oa_Ob = tamm.TensorInfo(
        [v_alpha, v_beta, o_alpha, o_beta],
        non_zero_check_Va_Vb_Oa_Ob,
    )

    d_r1 = tamm.TensorDouble([v_alpha, o_alpha], block_info_Va_Oa)
    d_r2 = tamm.TensorDouble([v_alpha, v_beta, o_alpha, o_beta], block_info_Va_Vb_Oa_Ob)

    tamm.TensorDouble.allocate(ec, d_r1, d_r2)

    d_t1 = tamm.TensorDouble([v_alpha, o_alpha], block_info_Va_Oa)
    d_t2 = tamm.TensorDouble([v_alpha, v_beta, o_alpha, o_beta], block_info_Va_Vb_Oa_Ob)

    tamm.TensorDouble.allocate(ec, d_t1, d_t2)

    return p_evl_sorted, d_t1, d_t2, d_r1, d_r2


# -----------------------------------------------------------------------------
# CCSD E
# -----------------------------------------------------------------------------

def ccsd_e_cs(sch, MO, CI, de, t1_aa, t2_abab, t2_aaaa, f1, chol3d):
    global _a01V, _a02_sp, _a03_sp, t2_aaaa_temp

    (cind,) = CI.labels("all", count=1)

    p1_va, p2_va = v_alpha.labels("all", count=2)
    (p1_vb,) = v_beta.labels("all", count=1)
    h1_oa, h2_oa = o_alpha.labels("all", count=2)
    (h1_ob,) = o_beta.labels("all", count=1)

    exact_copy(
        sch,
        t2_aaaa(p1_va, p2_va, h1_oa, h2_oa),
        t2_abab(p1_va, p1_vb, h1_oa, h1_ob),
        True,
    )

    sch(t2_aaaa_temp(), "=", 0.0)
    sch(t2_aaaa_temp(), "=", t2_aaaa())
    sch(t2_aaaa(p1_va, p2_va, h1_oa, h2_oa), "+=", -1.0 * t2_aaaa_temp(p2_va, p1_va, h1_oa, h2_oa))
    sch(t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa), "+=", t2_aaaa(p2_va, p1_va, h2_oa, h1_oa))

    sch(_a01V(cind), "=", t1_aa(p1_va, h1_oa) * chol3d(h1_oa, p1_va, cind))
    sch(_a02_sp(h1_oa, h2_oa, cind), "=", t1_aa(p1_va, h1_oa) * chol3d(h2_oa, p1_va, cind))
    sch(_a03_sp(h2_oa, p2_va, cind), "=", t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d(h1_oa, p1_va, cind))

    sch(de(), "=", 2.0 * _a01V() * _a01V())
    sch(de(), "+=", -1.0 * _a02_sp(h1_oa, h2_oa, cind) * _a02_sp(h2_oa, h1_oa, cind))
    sch(de(), "+=", _a03_sp(h1_oa, p1_va, cind) * chol3d(h1_oa, p1_va, cind))
    sch(de(), "+=", 2.0 * t1_aa(p1_va, h1_oa) * f1(h1_oa, p1_va))


# -----------------------------------------------------------------------------
# CCSD T1
# -----------------------------------------------------------------------------

def ccsd_t1_cs(sch, MO, CI, i0_aa, t1_aa, t2_abab, f1, chol3d):
    global _a01_sp, _a02V, _a04_sp, _a05_sp, _a06_sp, t2_aaaa_temp

    (cind,) = CI.labels("all", count=1)
    (p2,) = MO.labels("virt", count=1)
    (h1,) = MO.labels("occ", count=1)
    _ = (p2, h1)

    p1_va, p2_va = v_alpha.labels("all", count=2)
    (p1_vb,) = v_beta.labels("all", count=1)
    h1_oa, h2_oa = o_alpha.labels("all", count=2)
    (h1_ob,) = o_beta.labels("all", count=1)
    _ = (p1_vb, h1_ob)

    sch(i0_aa(p2_va, h1_oa), "=", f1(h1_oa, p2_va))
    sch(_a01_sp(h2_oa, h1_oa, cind), "=", t1_aa(p1_va, h1_oa) * chol3d(h2_oa, p1_va, cind))
    sch(_a02V(cind), "=", 2.0 * t1_aa(p1_va, h1_oa) * chol3d(h1_oa, p1_va, cind))

    sch(_a05_sp(h2_oa, p1_va), "=", -1.0 * chol3d(h1_oa, p1_va, cind) * _a01_sp(h2_oa, h1_oa, cind))
    sch(_a05_sp(h2_oa, p1_va), "+=", f1(h2_oa, p1_va))

    sch(_a06_sp(p1_va, h1_oa, cind), "=", -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d(h2_oa, p2_va, cind))

    sch(_a04_sp(h2_oa, h1_oa), "=", -1.0 * f1(h2_oa, h1_oa))
    sch(_a04_sp(h2_oa, h1_oa), "+=", chol3d(h2_oa, p1_va, cind) * _a06_sp(p1_va, h1_oa, cind))
    sch(_a04_sp(h2_oa, h1_oa), "+=", -1.0 * t1_aa(p1_va, h1_oa) * f1(h2_oa, p1_va))

    sch(i0_aa(p2_va, h1_oa), "+=", t1_aa(p2_va, h2_oa) * _a04_sp(h2_oa, h1_oa))
    sch(i0_aa(p1_va, h2_oa), "+=", chol3d(h2_oa, p1_va, cind) * _a02V(cind))
    sch(i0_aa(p1_va, h2_oa), "+=", t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05_sp(h1_oa, p2_va))
    sch(i0_aa(p2_va, h1_oa), "+=", -1.0 * chol3d(p2_va, p1_va, cind) * _a06_sp(p1_va, h1_oa, cind))

    sch(_a06_sp(p2_va, h2_oa, cind), "+=", -1.0 * t1_aa(p1_va, h2_oa) * chol3d(p2_va, p1_va, cind))
    sch(i0_aa(p1_va, h2_oa), "+=", -1.0 * _a06_sp(p1_va, h2_oa, cind) * _a02V(cind))

    sch(_a06_sp(p2_va, h1_oa, cind), "+=", -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind))
    sch(_a06_sp(p2_va, h1_oa, cind), "+=", t1_aa(p2_va, h2_oa) * _a01_sp(h2_oa, h1_oa, cind))

    sch(_a01_sp(h2_oa, h1_oa, cind), "+=", chol3d(h2_oa, h1_oa, cind))
    sch(i0_aa(p2_va, h1_oa), "+=", _a01_sp(h2_oa, h1_oa, cind) * _a06_sp(p2_va, h2_oa, cind))
    sch(i0_aa(p2_va, h1_oa), "+=", t1_aa(p1_va, h1_oa) * f1(p2_va, p1_va))


# -----------------------------------------------------------------------------
# CCSD T2
# -----------------------------------------------------------------------------

def ccsd_t2_cs(sch, MO, CI, i0_abab, t1_aa, t2_abab, t2_aaaa, f1, chol3d):
    global _a001_sp, _a004_sp, _a006_sp, _a007V, _a008_sp, _a009_sp
    global _a017_sp, _a019_sp, _a020_sp, _a021_sp, _a022_sp, i0_temp, t2_aaaa_temp

    (cind,) = CI.labels("all", count=1)
    p3, p4 = MO.labels("virt", count=2)
    h1, h2 = MO.labels("occ", count=2)
    _ = (p3, p4, h1, h2)

    p1_va, p2_va, p3_va = v_alpha.labels("all", count=3)
    p1_vb, p2_vb = v_beta.labels("all", count=2)
    h1_oa, h2_oa, h3_oa = o_alpha.labels("all", count=3)
    h1_ob, h2_ob = o_beta.labels("all", count=2)

    sch(_a017_sp(p1_va, h2_oa, cind), "=", -1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d(h1_oa, p2_va, cind))
    sch(_a006_sp(h2_oa, h1_oa), "=", -1.0 * chol3d(h2_oa, p2_va, cind) * _a017_sp(p2_va, h1_oa, cind))
    sch(_a007V(cind), "=", 2.0 * chol3d(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa))
    sch(_a009_sp(h1_oa, h2_oa, cind), "=", chol3d(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa))
    sch(_a021_sp(p2_va, p1_va, cind), "=", -0.5 * chol3d(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa))
    sch(_a021_sp(p2_va, p1_va, cind), "+=", 0.5 * chol3d(p2_va, p1_va, cind))
    sch(_a017_sp(p1_va, h2_oa, cind), "+=", -2.0 * t1_aa(p2_va, h2_oa) * _a021_sp(p1_va, p2_va, cind))
    sch(_a008_sp(h2_oa, h1_oa, cind), "=", _a009_sp(h2_oa, h1_oa, cind))
    sch(_a009_sp(h2_oa, h1_oa, cind), "+=", chol3d(h2_oa, h1_oa, cind))

    sch(_a001_sp(p1_va, p2_va), "=", -2.0 * _a021_sp(p1_va, p2_va, cind) * _a007V(cind))
    sch(_a001_sp(p1_va, p2_va), "+=", -1.0 * _a017_sp(p1_va, h2_oa, cind) * chol3d(h2_oa, p2_va, cind))
    sch(_a006_sp(h2_oa, h1_oa), "+=", _a009_sp(h2_oa, h1_oa, cind) * _a007V(cind))
    sch(_a006_sp(h3_oa, h1_oa), "+=", -1.0 * _a009_sp(h2_oa, h1_oa, cind) * _a008_sp(h3_oa, h2_oa, cind))

    sch(_a019_sp(h2_oa, h1_ob, h1_oa, h2_ob), "=", 0.25 * _a009_sp(h2_oa, h1_oa, cind) * _a009_sp(h1_ob, h2_ob, cind))
    sch(_a020_sp(p2_va, h2_oa, p1_va, h1_oa), "=", -2.0 * _a009_sp(h2_oa, h1_oa, cind) * _a021_sp(p2_va, p1_va, cind))

    sch(_a020_sp(p1_va, h3_oa, p3_va, h2_oa), "+=", 0.5 * _a004_sp(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va, p2_va, h1_oa, h2_oa))
    sch(_a020_sp(p1_vb, h2_oa, p1_va, h2_ob), "=", -0.5 * _a004_sp(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va, p1_vb, h1_oa, h2_ob))
    sch(_a020_sp(p1_vb, h1_oa, p2_vb, h2_oa), "+=", 0.5 * _a004_sp(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob))

    sch(_a017_sp(p1_va, h2_oa, cind), "+=", t1_aa(p1_va, h1_oa) * chol3d(h1_oa, h2_oa, cind))
    sch(_a017_sp(p1_va, h2_oa, cind), "+=", -1.0 * chol3d(h2_oa, p1_va, cind))
    sch(_a001_sp(p2_va, p1_va), "+=", -1.0 * f1(p2_va, p1_va))
    sch(_a001_sp(p2_va, p1_va), "+=", t1_aa(p2_va, h1_oa) * f1(h1_oa, p1_va))
    sch(_a006_sp(h2_oa, h1_oa), "+=", f1(h2_oa, h1_oa))
    sch(_a006_sp(h2_oa, h1_oa), "+=", t1_aa(p1_va, h1_oa) * f1(h2_oa, p1_va))

    sch(i0_abab(p1_va, p2_vb, h2_oa, h1_ob), "=", _a020_sp(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob))
    sch(i0_abab(p2_va, p1_vb, h2_oa, h1_ob), "+=", _a020_sp(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa))
    sch(i0_abab(p1_va, p1_vb, h2_oa, h1_ob), "+=", _a020_sp(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob))
    sch(i0_abab(p1_va, p1_vb, h2_oa, h1_ob), "+=", i0_temp(p1_vb, p1_va, h1_ob, h2_oa))
    sch(i0_abab(p1_va, p1_vb, h1_oa, h2_ob), "+=", _a017_sp(p1_va, h1_oa, cind) * _a017_sp(p1_vb, h2_ob, cind))
    sch(_a022_sp(p1_va, p2_vb, p2_va, p1_vb), "=", _a021_sp(p1_va, p2_va, cind) * _a021_sp(p2_vb, p1_vb, cind))
    sch(i0_abab(p1_va, p2_vb, h1_oa, h2_ob), "+=", 4.0 * _a022_sp(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va, p1_vb, h1_oa, h2_ob))
    sch(_a019_sp(h2_oa, h1_ob, h1_oa, h2_ob), "+=", 0.25 * _a004_sp(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va, p2_vb, h1_oa, h2_ob))
    sch(i0_abab(p1_va, p1_vb, h1_oa, h2_ob), "+=", 4.0 * _a019_sp(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob))
    sch(i0_abab(p1_va, p1_vb, h1_oa, h2_ob), "+=", -1.0 * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001_sp(p1_vb, p2_vb))
    sch(i0_abab(p1_va, p1_vb, h1_oa, h2_ob), "+=", -1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001_sp(p1_va, p2_va))
    sch(i0_abab(p1_va, p1_vb, h2_oa, h1_ob), "+=", -1.0 * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006_sp(h1_oa, h2_oa))
    sch(i0_abab(p1_va, p1_vb, h1_oa, h2_ob), "+=", -1.0 * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006_sp(h1_ob, h2_ob))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv=None):
    global o_alpha, v_alpha, o_beta, v_beta
    global _a01V, _a02V, _a007V
    global _a01_sp, _a02_sp, _a03_sp, _a04_sp, _a05_sp, _a06_sp
    global _a001_sp, _a004_sp, _a006_sp, _a008_sp, _a009_sp, _a017_sp
    global _a019_sp, _a020_sp, _a021_sp, _a022_sp
    global i0_temp, t2_aaaa_temp

    if argv is None:
        argv = sys.argv

    tamm.initialize(argv, False)

    try:
        if len(argv) < 5:
            raise RuntimeError("Please provide occ_alpha, virt_alpha, cholesky-count and tile size")

        n_occ_alpha = atoi_cpp(argv[1])
        n_vir_alpha = atoi_cpp(argv[2])
        chol_count = atoi_cpp(argv[3])
        tile_size = atoi_cpp(argv[4])

        nbf = n_occ_alpha + n_vir_alpha

        pg = tamm.ProcGroup.create_world_coll()
        ec = tamm.ExecutionContext(pg, tamm.DistributionKind.nw, tamm.MemoryManagerKind.ga)
        exhw = ec.exhw()
        sch = tamm.Scheduler(ec)
        profile = True

        print_header(ec, nbf, n_occ_alpha, n_vir_alpha, chol_count, tile_size)

        n_occ_beta = n_occ_alpha
        n_vir_beta = n_vir_alpha
        nmo = 2 * nbf
        nocc = 2 * n_occ_alpha

        MO = make_mo(n_occ_alpha, n_vir_alpha, tile_size)

        N = MO("all")

        is_non_zero_2D_spin = generate_spin_check(
            [N, N],
            [tamm.SpinPosition.lower, tamm.SpinPosition.upper],
        )

        tensor_info_N_N = tamm.TensorInfo([N, N], is_non_zero_2D_spin)

        d_f1 = tamm.TensorDouble([N, N], tensor_info_N_N)
        tamm.TensorDouble.allocate(ec, d_f1)
        tamm.random_ip(d_f1)

        p_evl_sorted, t1_aa, t2_abab, r1_aa, r2_abab = setupTensors_cs(ec, MO, d_f1)

        chol_is = tamm.IndexSpace(tamm.range(0, chol_count))
        CI = tamm.TiledIndexSpace(chol_is, 1000)

        O = MO("occ")
        V = MO("virt")

        (cind,) = CI.labels("all", count=1)

        otiles = int(O.num_tiles())
        vtiles = int(V.num_tiles())
        oatiles = int(MO("occ_alpha").num_tiles())
        vatiles = int(MO("virt_alpha").num_tiles())

        o_alpha = tamm.TiledIndexSpace(MO("occ"), tamm.range(oatiles))
        v_alpha = tamm.TiledIndexSpace(MO("virt"), tamm.range(vatiles))
        o_beta = tamm.TiledIndexSpace(MO("occ"), tamm.range(oatiles, otiles))
        v_beta = tamm.TiledIndexSpace(MO("virt"), tamm.range(vatiles, vtiles))

        p1_va, p2_va = v_alpha.labels("all", count=2)
        p1_vb, p2_vb = v_beta.labels("all", count=2)
        h3_oa, h4_oa = o_alpha.labels("all", count=2)
        h3_ob, h4_ob = o_beta.labels("all", count=2)

        d_e = tamm.TensorDouble()

        is_nonzero_Va_Va_Oa_Oa = generate_spin_check(
            [v_alpha, v_alpha, o_alpha, o_alpha],
            [
                tamm.SpinPosition.lower,
                tamm.SpinPosition.lower,
                tamm.SpinPosition.upper,
                tamm.SpinPosition.upper,
            ],
        )

        tensor_info_Va_Va_Oa_Oa = tamm.TensorInfo(
            [v_alpha, v_alpha, o_alpha, o_alpha],
            is_nonzero_Va_Va_Oa_Oa,
        )

        t2_aaaa = tamm.TensorDouble(
            [v_alpha, v_alpha, o_alpha, o_alpha],
            tensor_info_Va_Va_Oa_Oa,
        )

        tensor_info_f1 = tamm.TensorInfo(
            [MO, MO],
            ["IJ", "IA", "AB"],
            index_to_sub_string,
        )

        f1 = tamm.TensorDouble([MO, MO], tensor_info_f1)

        tensor_info_chol3d = tamm.TensorInfo(
            [MO, MO, CI],
            ["IJX", "IAX", "ABX"],
            index_to_sub_string,
        )

        chol3d = tamm.TensorDouble([MO, MO, CI], tensor_info_chol3d)

        _a01V = tamm.TensorDouble([CI])

        _a02_sp = tamm.TensorDouble([MO, MO, CI], [[h3_oa, h4_oa, cind]])
        _a03_sp = tamm.TensorDouble([MO, MO, CI], [[h3_oa, p1_va, cind]])

        _a004_sp = tamm.TensorDouble(
            [MO, MO, MO, MO],
            [
                "virt_alpha, virt_alpha, occ_alpha, occ_alpha",
                "virt_alpha, virt_beta, occ_alpha, occ_beta",
            ],
        )

        t2_aaaa_temp = tamm.TensorDouble([v_alpha, v_alpha, o_alpha, o_alpha])
        i0_temp = tamm.TensorDouble([v_beta, v_alpha, o_beta, o_alpha])

        _a02V = tamm.TensorDouble([CI])

        _a01_sp = tamm.TensorDouble([MO, MO, CI], [[h3_oa, h4_oa, cind]])
        _a04_sp = tamm.TensorDouble([MO, MO], [[h3_oa, h4_oa]])
        _a05_sp = tamm.TensorDouble([MO, MO], [[h3_oa, p1_va]])
        _a06_sp = tamm.TensorDouble([MO, MO, CI], [[p1_va, h3_oa, cind]])

        _a007V = tamm.TensorDouble([CI])

        _a001_sp = tamm.TensorDouble([MO, MO], [[p1_va, p2_va], [p1_vb, p2_vb]])
        _a006_sp = tamm.TensorDouble([MO, MO], [[h3_oa, h4_oa], [h3_ob, h4_ob]])
        _a008_sp = tamm.TensorDouble([MO, MO, CI], [[h3_oa, h4_oa, cind]])
        _a009_sp = tamm.TensorDouble([MO, MO, CI], [[h3_oa, h4_oa, cind], [h3_ob, h4_ob, cind]])
        _a017_sp = tamm.TensorDouble([MO, MO, CI], [[p1_va, h3_oa, cind], [p1_vb, h3_ob, cind]])
        _a021_sp = tamm.TensorDouble([MO, MO, CI], [[p1_va, p2_va, cind], [p1_vb, p2_vb, cind]])
        _a019_sp = tamm.TensorDouble([MO, MO, MO, MO], [[h3_oa, h3_ob, h4_oa, h4_ob]])
        _a022_sp = tamm.TensorDouble([MO, MO, MO, MO], [[p1_va, p1_vb, p2_va, p2_vb]])

        _a020_sp = tamm.TensorDouble(
            [MO, MO, MO, MO],
            [
                [p1_va, h3_oa, p2_va, h4_oa],
                [p1_vb, h3_oa, p2_vb, h4_oa],
                [p1_vb, h3_oa, p2_va, h4_ob],
                [p1_vb, h3_ob, p2_vb, h4_ob],
            ],
        )

        sch.allocate(t2_aaaa)
        sch.allocate(d_e, i0_temp, t2_aaaa_temp, _a01V)
        sch.allocate(f1, chol3d)
        sch.allocate(_a02_sp, _a03_sp)
        sch.allocate(_a02V, _a007V)
        sch.allocate(
            _a004_sp,
            _a01_sp,
            _a04_sp,
            _a05_sp,
            _a06_sp,
            _a001_sp,
            _a006_sp,
            _a008_sp,
            _a009_sp,
            _a017_sp,
            _a019_sp,
            _a020_sp,
            _a021_sp,
            _a022_sp,
        )
        sch.execute()

        tamm.random_ip(f1(h3_oa, h4_oa))
        tamm.random_ip(f1(h3_oa, p2_va))
        tamm.random_ip(f1(p1_va, p2_va))

        tamm.random_ip(chol3d(h3_oa, h4_oa, cind))
        tamm.random_ip(chol3d(h3_oa, p2_va, cind))
        tamm.random_ip(chol3d(p1_va, p2_va, cind))

        sch(
            _a004_sp(p1_va, p2_va, h4_oa, h3_oa),
            "=",
            chol3d(h4_oa, p1_va, cind) * chol3d(h3_oa, p2_va, cind),
        )

        exact_copy(
            sch,
            _a004_sp(p1_va, p1_vb, h3_oa, h3_ob),
            _a004_sp(p1_va, p2_va, h3_oa, h4_oa),
            True,
        )

        sch.execute(exhw)

        timer_start = time.perf_counter()

        ccsd_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1, chol3d)
        ccsd_t1_cs(sch, MO, CI, r1_aa, t1_aa, t2_abab, f1, chol3d)
        ccsd_t2_cs(sch, MO, CI, r2_abab, t1_aa, t2_abab, t2_aaaa, f1, chol3d)

        sch.execute(exhw, profile)

        timer_end = time.perf_counter()
        iter_time = timer_end - timer_start

        if ec.print():
            print(f"Time taken for closed-shell CD-CCSD: {iter_time}")

        if profile and ec.print():
            profile_csv = (
                f"ccsd_profile_{nbf}bf_{n_occ_alpha}oa_{n_vir_alpha}va_"
                f"{chol_count}cv_{tile_size}TS.csv"
            )
            with open(profile_csv, "w") as pds:
                pds.write(str(ec.get_profile_header()))
                pds.write("\n")
                pds.write(str(ec.get_profile_data()))
                pds.write("\n")

        sch.deallocate(_a02V, _a007V)
        sch.deallocate(
            _a004_sp,
            _a01_sp,
            _a04_sp,
            _a05_sp,
            _a06_sp,
            _a001_sp,
            _a006_sp,
            _a008_sp,
            _a009_sp,
            _a017_sp,
            _a019_sp,
            _a020_sp,
            _a021_sp,
            _a022_sp,
        )
        sch.deallocate(d_e, i0_temp, t2_aaaa_temp, _a01V)
        sch.deallocate(_a02_sp, _a03_sp)
        sch.deallocate(f1, chol3d)
        sch.execute()

        sch.deallocate(t1_aa, t2_abab, r1_aa, r2_abab, d_f1, t2_aaaa).execute()

    finally:
        tamm.finalize(True)


if __name__ == "__main__":
    main()
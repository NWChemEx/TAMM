import sys
import builtins
import pytamm as pt

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
trange = pt.range

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
#
# Important:
# The current binding uses py::arg("start") = make_label() for
# TiledIndexSpace.labels/label. In pybind11 that default is evaluated at module
# initialization, not at each Python call. To avoid accidental label reuse, this
# script explicitly supplies fresh integer labels whenever the C++ code used the
# default make_label().
#
_label_counter = 100000


def fresh_label_start(count=1):
    global _label_counter
    s = _label_counter
    _label_counter += int(count) + 17
    return s


def labels(tis, count=1, id="all", start=None):
    if start is None:
        start = fresh_label_start(count)
    return tuple(tis.labels(id, start, count))


def label(tis, id="all", start=None):
    if start is None:
        start = fresh_label_start(1)
    return tis.label(id, start)


def size_vec(*xs):
    try:
        v = pt.SizeVec()
        for x in xs:
            v.append(int(x))
        return v
    except Exception:
        return [int(x) for x in xs]


def spin_mask(*positions):
    try:
        sm = pt.SpinMask()
        for p in positions:
            sm.append(p)
        return sm
    except Exception:
        return list(positions)


# ---------------------------------------------------------------------------
# TAMM helpers
# ---------------------------------------------------------------------------

def make_ec():
    pg = pt.ProcGroup.create_world_coll()
    ec = pt.ExecutionContext(pg, pt.DistributionKind.nw, pt.MemoryManagerKind.ga)
    return pg, ec


def rank0():
    try:
        return int(pt.ProcGroup.world_rank()) == 0
    except Exception:
        return True


def rprint(*args, **kwargs):
    if rank0():
        print(*args, **kwargs)


def maybe_print_tensor(tensor, title=None):
    if VERBOSE:
        if title and rank0():
            print(title)
        pt.print_tensor(tensor)


def dealloc(*tensors):
    for t in tensors:
        if t is None:
            continue
        try:
            if t.is_allocated():
                t.deallocate()
        except Exception:
            pass


def check_value(t_or_lt, val, tol=1.0e-10):
    """
    Python version of the C++ check_value(Tensor/LabeledTensor, val).

    For every block in the labeled loop nest:
      - expected value is val if tensor.is_non_zero(blockid)
      - otherwise expected value is 0
    """
    if hasattr(t_or_lt, "labels") and hasattr(t_or_lt, "tensor"):
        lt = t_or_lt
    else:
        lt = t_or_lt()

    tensor = lt.tensor()

    for itval in pt.LabelLoopNest(lt.labels()):
        blockid = pt.translate_blockid(itval, lt)
        bsize = int(tensor.block_size(blockid))
        buf = [0.0] * bsize
        tensor.get(blockid, buf)

        ref = val if tensor.is_non_zero(blockid) else 0.0
        for i, x in enumerate(buf):
            assert abs(x - ref) < tol, (
                f"check_value failed at block {tuple(blockid)}, elem {i}: "
                f"got {x}, expected {ref}"
            )


def lambda_function(blockid, buff):
    buff[...] = 42.0


def make_l_func(last_idx):
    def f(blockid, buf):
        if blockid[0] == last_idx or blockid[1] == last_idx:
            buf[...] = -1.0
        else:
            buf[...] = 0.0

        if blockid[0] == last_idx and blockid[1] == last_idx:
            buf[...] = 0.0

    return f


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_block_sparse_tensor_construction():
    SpinIS = pt.IndexSpace(
        trange(0, 20),
        {
            "occ": [trange(0, 10)],
            "virt": [trange(10, 20)],
        },
        {
            pt.Spin(1): [trange(0, 5), trange(10, 15)],
            pt.Spin(-1): [trange(5, 10), trange(15, 20)],
        },
    )

    IS = pt.IndexSpace(trange(0, 20))
    SpinTIS = pt.TiledIndexSpace(SpinIS, 5)
    TIS = pt.TiledIndexSpace(IS, 5)

    spin_positions_2d = [pt.SpinPosition.lower, pt.SpinPosition.upper]
    spin_mask_2d = spin_mask(*spin_positions_2d)

    pg, ec = make_ec()

    # -----------------------------------------------------------------------
    # TensorInfo from Python NonZeroCheck: spin conservation
    # -----------------------------------------------------------------------
    t_spaces = [SpinTIS, SpinTIS]

    def is_non_zero_spin(blockid):
        upper_total = 0
        lower_total = 0
        other_total = 0

        for idx in builtins.range(2):
            s = int(t_spaces[idx].spin(blockid[idx]))
            if spin_positions_2d[idx] == pt.SpinPosition.upper:
                upper_total += s
            elif spin_positions_2d[idx] == pt.SpinPosition.lower:
                lower_total += s
            else:
                other_total += s

        return upper_total == lower_total

    tensor_info = pt.TensorInfo(t_spaces, is_non_zero_spin)
    tensor = pt.TensorDouble(t_spaces, tensor_info)
    tensor.allocate_self(ec)

    sch = pt.Scheduler(ec)
    sch(tensor(), "=", 42.0).execute()
    check_value(tensor, 42.0)
    tensor.deallocate()

    # -----------------------------------------------------------------------
    # TensorInfo from Python NonZeroCheck: diagonal blocks only
    # -----------------------------------------------------------------------
    def is_non_zero_diag(blockid):
        return blockid[0] == blockid[1]

    tensor_info = pt.TensorInfo(t_spaces, is_non_zero_diag)
    tensor = pt.TensorDouble(t_spaces, tensor_info)
    tensor.allocate_self(ec)

    sch = pt.Scheduler(ec)
    sch(tensor(), "=", 42.0).execute()
    check_value(tensor, 42.0)
    tensor.deallocate()

    # -----------------------------------------------------------------------
    # Block sparse MO examples
    # -----------------------------------------------------------------------
    MO_IS = pt.IndexSpace(
        trange(0, 20),
        {
            "occ": [trange(0, 10)],
            "virt": [trange(10, 20)],
        },
    )
    MO = pt.TiledIndexSpace(MO_IS, 5)

    i, j, k, l = labels(MO("occ"), 4)
    a, b, c, d = labels(MO("virt"), 4)

    char2MOstr = {
        "i": "occ",
        "j": "occ",
        "k": "occ",
        "l": "occ",
        "a": "virt",
        "b": "virt",
        "c": "virt",
        "d": "virt",
    }

    tensor_info = pt.TensorInfo(
        [MO, MO, MO, MO],
        ["ijab", "iajb", "ijka", "ijkl", "iabc", "abcd"],
        char2MOstr,
        ["abij", "aibj"],
    )

    tensor = pt.TensorDouble([MO, MO, MO, MO], tensor_info)
    tensor.allocate_self(ec)

    sch = pt.Scheduler(ec)
    sch(tensor(), "=", 42.0).execute()
    check_value(tensor, 42.0)

    sch = pt.Scheduler(ec)
    sch(tensor(i, j, a, b), "=", 1.0)
    sch(tensor(i, a, j, b), "=", 2.0)
    sch(tensor(i, j, k, a), "=", 3.0)
    sch(tensor(i, j, k, l), "=", 4.0)
    sch(tensor(i, a, b, c), "=", 5.0)
    sch(tensor(a, b, c, d), "=", 6.0)
    sch.execute()

    check_value(tensor(i, j, a, b), 1.0)
    check_value(tensor(i, a, j, b), 2.0)
    check_value(tensor(i, j, k, a), 3.0)
    check_value(tensor(i, j, k, l), 4.0)
    check_value(tensor(i, a, b, c), 5.0)
    check_value(tensor(a, b, c, d), 6.0)
    tensor.deallocate()

    # -----------------------------------------------------------------------
    # Tensor ctor: allowed strings + char map
    # -----------------------------------------------------------------------
    tensor = pt.TensorDouble([MO, MO, MO, MO], ["ijab", "ijka", "iajb"], char2MOstr)
    tensor.allocate_self(ec)

    sch = pt.Scheduler(ec)
    sch(tensor(), "=", 42.0).execute()
    check_value(tensor, 42.0)

    sch = pt.Scheduler(ec)
    sch(tensor(i, j, a, b), "=", 1.0)
    sch(tensor(i, a, j, b), "=", 2.0)
    sch(tensor(i, j, k, a), "=", 3.0)
    sch.execute()

    check_value(tensor(i, j, a, b), 1.0)
    check_value(tensor(i, a, j, b), 2.0)
    check_value(tensor(i, j, k, a), 3.0)
    tensor.deallocate()

    # -----------------------------------------------------------------------
    # Three sparse tensors using allowed strings + char map
    # -----------------------------------------------------------------------
    tensorA = pt.TensorDouble([MO, MO, MO, MO], ["ijab", "ijkl"], char2MOstr)
    tensorB = pt.TensorDouble([MO, MO, MO, MO], ["ijka", "iajb"], char2MOstr)
    tensorC = pt.TensorDouble([MO, MO, MO, MO], ["iabc", "abcd"], char2MOstr)

    tensorA.allocate_self(ec)
    tensorB.allocate_self(ec)
    tensorC.allocate_self(ec)

    sch = pt.Scheduler(ec)
    sch(tensorA(), "=", 2.0)
    sch(tensorB(), "=", 4.0)
    sch(tensorC(), "=", 0.0)
    sch.execute()

    check_value(tensorA, 2.0)
    check_value(tensorB, 4.0)
    check_value(tensorC, 0.0)

    sch = pt.Scheduler(ec)
    sch(tensorC(a, b, c, d), "+=", tensorA(i, j, a, b) * tensorB(j, c, i, d))
    sch(tensorC(i, a, b, c), "+=", 0.5 * tensorA(j, k, a, b) * tensorB(i, j, k, c))
    sch.execute()

    check_value(tensorC(i, a, b, c), 400.0)
    check_value(tensorC(a, b, c, d), 800.0)
    dealloc(tensorA, tensorB, tensorC)

    # -----------------------------------------------------------------------
    # Tensor ctor: allowed IndexLabelVecs
    # -----------------------------------------------------------------------
    tensorA = pt.TensorDouble([MO, MO, MO, MO], [[i, j, a, b], [i, j, k, l]])
    tensorB = pt.TensorDouble([MO, MO, MO, MO], [[i, j, k, a], [i, a, j, b]])
    tensorC = pt.TensorDouble([MO, MO, MO, MO], [[i, a, b, c], [a, b, c, d]])

    tensorA.allocate_self(ec)
    tensorB.allocate_self(ec)
    tensorC.allocate_self(ec)

    sch = pt.Scheduler(ec)
    sch(tensorA(), "=", 2.0)
    sch(tensorB(), "=", 4.0)
    sch(tensorC(), "=", 0.0)
    sch.execute()

    check_value(tensorA, 2.0)
    check_value(tensorB, 4.0)
    check_value(tensorC, 0.0)

    sch = pt.Scheduler(ec)
    sch(tensorC(a, b, c, d), "+=", tensorA(i, j, a, b) * tensorB(j, c, i, d))
    sch(tensorC(i, a, b, c), "+=", 0.5 * tensorA(j, k, a, b) * tensorB(i, j, k, c))
    sch.execute()

    check_value(tensorC(i, a, b, c), 400.0)
    check_value(tensorC(a, b, c, d), 800.0)
    dealloc(tensorA, tensorB, tensorC)

    # -----------------------------------------------------------------------
    # Tensor ctor: allowed named-subspace strings
    # -----------------------------------------------------------------------
    tensorA = pt.TensorDouble(
        [MO, MO, MO, MO],
        ["occ, occ, virt, virt", "occ, occ, occ, occ"],
    )
    tensorB = pt.TensorDouble(
        [MO, MO, MO, MO],
        ["occ, occ, occ, virt", "occ, virt, occ, virt"],
    )
    tensorC = pt.TensorDouble(
        [MO, MO, MO, MO],
        ["occ, virt, virt, virt", "virt, virt, virt, virt"],
    )

    tensorA.allocate_self(ec)
    tensorB.allocate_self(ec)
    tensorC.allocate_self(ec)

    sch = pt.Scheduler(ec)
    sch(tensorA(), "=", 2.0)
    sch(tensorB(), "=", 4.0)
    sch(tensorC(), "=", 0.0)
    sch.execute()

    check_value(tensorA, 2.0)
    check_value(tensorB, 4.0)
    check_value(tensorC, 0.0)

    sch = pt.Scheduler(ec)
    sch(tensorC(a, b, c, d), "+=", tensorA(i, j, a, b) * tensorB(j, c, i, d))
    sch(tensorC(i, a, b, c), "+=", 0.5 * tensorA(j, k, a, b) * tensorB(i, j, k, c))
    sch.execute()

    check_value(tensorC(i, a, b, c), 400.0)
    check_value(tensorC(a, b, c, d), 800.0)
    dealloc(tensorA, tensorB, tensorC)

    # -----------------------------------------------------------------------
    # Tensor ctor: allowed TiledIndexSpaceVecs
    # -----------------------------------------------------------------------
    Occ = MO("occ")
    Virt = MO("virt")

    tensorA = pt.TensorDouble(
        [MO, MO, MO, MO],
        [[Occ, Occ, Virt, Virt], [Occ, Occ, Occ, Occ]],
    )
    tensorB = pt.TensorDouble(
        [MO, MO, MO, MO],
        [[Occ, Occ, Occ, Virt], [Occ, Virt, Occ, Virt]],
    )
    tensorC = pt.TensorDouble(
        [MO, MO, MO, MO],
        [[Occ, Virt, Virt, Virt], [Virt, Virt, Virt, Virt]],
    )

    tensorA.allocate_self(ec)
    tensorB.allocate_self(ec)
    tensorC.allocate_self(ec)

    sch = pt.Scheduler(ec)
    sch(tensorA(), "=", 2.0)
    sch(tensorB(), "=", 4.0)
    sch(tensorC(), "=", 0.0)
    sch.execute()

    check_value(tensorA, 2.0)
    check_value(tensorB, 4.0)
    check_value(tensorC, 0.0)

    sch = pt.Scheduler(ec)
    sch(tensorC(a, b, c, d), "+=", tensorA(i, j, a, b) * tensorB(j, c, i, d))
    sch(tensorC(i, a, b, c), "+=", 0.5 * tensorA(j, k, a, b) * tensorB(i, j, k, c))
    sch.execute()

    check_value(tensorC(i, a, b, c), 400.0)
    check_value(tensorC(a, b, c, d), 800.0)
    dealloc(tensorA, tensorB, tensorC)


def test_spin_tensor_construction():
    SpinIS = pt.IndexSpace(
        trange(0, 20),
        {
            "occ": [trange(0, 10)],
            "virt": [trange(10, 20)],
        },
        {
            pt.Spin(1): [trange(0, 5), trange(10, 15)],
            pt.Spin(-1): [trange(5, 10), trange(15, 20)],
        },
    )
    IS = pt.IndexSpace(trange(0, 20))

    SpinTIS = pt.TiledIndexSpace(SpinIS, 5)
    TIS = pt.TiledIndexSpace(IS, 5)

    spin_mask_2d = spin_mask(pt.SpinPosition.lower, pt.SpinPosition.upper)

    i, j = labels(SpinTIS, 2)
    k, l = labels(TIS, 2)

    # Construction checks
    _ = pt.TensorDouble([SpinTIS, SpinTIS], spin_mask_2d)
    _ = pt.TensorDouble([i, j], spin_mask_2d)
    _ = pt.TensorDouble([TIS, TIS], spin_mask_2d)

    assert SpinTIS.spin(0) == pt.Spin(1)
    assert SpinTIS.spin(1) == pt.Spin(-1)
    assert SpinTIS.spin(2) == pt.Spin(1)
    assert SpinTIS.spin(3) == pt.Spin(-1)
    assert SpinTIS("occ").spin(0) == pt.Spin(1)
    assert SpinTIS("occ").spin(1) == pt.Spin(-1)
    assert SpinTIS("virt").spin(0) == pt.Spin(1)
    assert SpinTIS("virt").spin(1) == pt.Spin(-1)

    tis_3 = pt.TiledIndexSpace(SpinIS, 3)

    expected_all = [1, 1, -1, -1, 1, 1, -1]
    for idx, sp in enumerate(expected_all):
        assert tis_3.spin(idx) == pt.Spin(sp)

    expected_sub = [1, 1, -1, -1]
    for idx, sp in enumerate(expected_sub):
        assert tis_3("occ").spin(idx) == pt.Spin(sp)
        assert tis_3("virt").spin(idx) == pt.Spin(sp)

    pg, ec = make_ec()

    tensor = pt.TensorDouble([tis_3, tis_3], spin_mask_2d)
    tensor.allocate_self(ec)
    pt.Scheduler(ec)(tensor(), "=", 42.0).execute()
    check_value(tensor, 42.0)
    tensor.deallocate()

    tensor = pt.TensorDouble([tis_3("occ"), tis_3("virt")], spin_mask_2d)
    tensor.allocate_self(ec)
    pt.Scheduler(ec)(tensor(), "=", 42.0).execute()
    check_value(tensor, 42.0)
    tensor.deallocate()

    T1 = pt.TensorDouble([tis_3, tis_3], spin_mask_2d)
    T2 = pt.TensorDouble([tis_3, tis_3], spin_mask_2d)
    T1.allocate_self(ec)
    T2.allocate_self(ec)
    sch = pt.Scheduler(ec)
    sch(T2(), "=", 3.0)
    sch(T1(), "=", T2())
    sch.execute()
    check_value(T2, 3.0)
    check_value(T1, 3.0)
    dealloc(T1, T2)

    T1 = pt.TensorDouble([tis_3, tis_3], size_vec(1, 1))
    T2 = pt.TensorDouble([tis_3, tis_3], size_vec(1, 1))
    T1.allocate_self(ec)
    T2.allocate_self(ec)
    sch = pt.Scheduler(ec)
    sch(T1(), "=", 42.0)
    sch(T2(), "=", 3.0)
    sch(T1(), "+=", T2())
    sch.execute()
    check_value(T2, 3.0)
    check_value(T1, 45.0)
    dealloc(T1, T2)

    T1 = pt.TensorDouble([tis_3, tis_3], size_vec(1, 1))
    T2 = pt.TensorDouble([tis_3, tis_3], size_vec(1, 1))
    T1.allocate_self(ec)
    T2.allocate_self(ec)
    sch = pt.Scheduler(ec)
    sch(T1(), "=", 42.0)
    sch(T2(), "=", 3.0)
    sch(T1(), "+=", 2.0 * T2())
    sch.execute()
    check_value(T2, 3.0)
    check_value(T1, 48.0)
    dealloc(T1, T2)

    T1 = pt.TensorDouble([tis_3, tis_3], size_vec(1, 1))
    T2 = pt.TensorDouble([tis_3, tis_3], size_vec(1, 1))
    T3 = pt.TensorDouble([tis_3, tis_3], size_vec(1, 1))
    T1.allocate_self(ec)
    T2.allocate_self(ec)
    T3.allocate_self(ec)

    # C++: Tensor<T> T4 = T3; Tensor copy is a handle copy.
    T4 = T3

    sch = pt.Scheduler(ec)
    sch(T1(), "=", 42.0)
    sch(T2(), "=", 3.0)
    sch(T3(), "=", 4.0)
    sch(T1(), "+=", T4() * T2())
    sch.execute()

    check_value(T3, 4.0)
    check_value(T2, 3.0)
    check_value(T1, 54.0)
    dealloc(T1, T2, T3)

    # Tensor backed by lambda/function object, direct get
    def local_lambda(blockid, buff):
        buff[...] = 42.0

    t = pt.TensorDouble([TIS, TIS], local_lambda)
    check_value(t, 42.0)

    # Lambda tensor in scheduler expression
    S = pt.TensorDouble([TIS, TIS], local_lambda)
    T1 = pt.TensorDouble([TIS, TIS])
    T1.allocate_self(ec)
    sch = pt.Scheduler(ec)
    sch(T1(), "=", 0.0)
    sch(T1(), "+=", 2.0 * S())
    sch.execute()
    check_value(T1, 84.0)
    dealloc(T1)

    # Function pointer analogue
    S = pt.TensorDouble([TIS, TIS], lambda_function)
    T1 = pt.TensorDouble([TIS, TIS])
    T1.allocate_self(ec)
    sch = pt.Scheduler(ec)
    sch(T1(), "=", 0.0)
    sch(T1(), "+=", 2.0 * S())
    sch.execute()
    check_value(T1, 84.0)
    dealloc(T1)

    # Vector of tensors allocation
    x1 = []
    x2 = []
    for _idx in builtins.range(5):
        x1.append(pt.TensorDouble([TIS, TIS]))
        x2.append(pt.TensorDouble([TIS, TIS]))
        pt.TensorDouble.allocate(ec, x1[-1], x2[-1])
    for t in x1 + x2:
        t.deallocate()

    # Uneven tiles compatibility/copy/add test
    MO_IS = pt.IndexSpace(trange(0, 7))
    MO = pt.TiledIndexSpace(MO_IS, [1, 1, 3, 1, 1])

    MO_IS2 = pt.IndexSpace(trange(0, 7))
    MO2 = pt.TiledIndexSpace(MO_IS2, [1, 1, 3, 1, 1])

    pT = pt.TensorDouble([MO, MO])
    pV = pt.TensorDouble([MO2, MO2])
    pT.allocate_self(ec)
    pV.allocate_self(ec)

    tis_list = pT.tiled_index_spaces()
    H = pt.TensorDouble(tis_list)
    H.allocate_self(ec)

    sch = pt.Scheduler(ec)
    sch(H("mu", "nu"), "=", pT("mu", "nu"))
    sch(H("mu", "nu"), "+=", pV("mu", "nu"))
    sch.execute()

    dealloc(pT, pV, H)

    # Simple allocate/deallocate with local EC
    IS10 = pt.IndexSpace(trange(10))
    TIS10 = pt.TiledIndexSpace(IS10, 2)
    A = pt.TensorDouble([TIS10, TIS10])
    pg2, ec2 = make_ec()
    A.allocate_self(ec2)
    A.deallocate()

    # Density-like contraction example
    AO_IS = pt.IndexSpace(trange(10))
    AO = pt.TiledIndexSpace(AO_IS, 2)
    MO_IS = pt.IndexSpace(trange(10))
    MO = pt.TiledIndexSpace(MO_IS, 2)

    C = pt.TensorDouble([AO, MO])
    pg3, ec3 = make_ec()
    sch = pt.Scheduler(ec3)
    sch.allocate(C).execute()

    AOs = C.tiled_index_spaces()[0]
    MOs = C.tiled_index_spaces()[1]
    mu, nu = labels(AOs, 2)
    (p,) = labels(MOs, 1)

    rho = pt.TensorDouble([AOs, AOs])
    sch.allocate(rho)
    sch(rho(), "=", 0.0)
    sch(rho(mu, nu), "+=", C(mu, p) * C(nu, p))
    sch.execute()

    dealloc(rho, C)


def test_hash_based_equality_and_compatibility_check():
    is1 = pt.IndexSpace(
        trange(0, 20),
        {
            "occ": [trange(0, 10)],
            "virt": [trange(10, 20)],
        },
    )
    is2 = pt.IndexSpace(trange(0, 10))
    is1_occ = is1("occ")

    tis1 = pt.TiledIndexSpace(is1)
    tis2 = pt.TiledIndexSpace(is2)
    tis3 = pt.TiledIndexSpace(is1_occ)
    sub_tis1 = pt.TiledIndexSpace(tis1, trange(0, 10))

    assert tis2 == tis3
    assert tis2 == tis1("occ")
    assert tis3 == tis1("occ")
    assert tis1 != tis2
    assert tis1 != tis3
    assert tis2 != tis1("virt")
    assert tis3 != tis1("virt")

    assert sub_tis1 == tis2
    assert sub_tis1 == tis3
    assert sub_tis1 == tis1("occ")
    assert sub_tis1 != tis1
    assert sub_tis1 != tis1("virt")

    assert sub_tis1.is_compatible_with(tis1)
    assert sub_tis1.is_compatible_with(tis1("occ"))
    assert not sub_tis1.is_compatible_with(tis2)
    assert not sub_tis1.is_compatible_with(tis3)
    assert sub_tis1.is_compatible_with(tis1("virt"))


def test_github_issues():
    pg, ec = make_ec()

    X = pt.TiledIndexSpace(pt.IndexSpace(trange(0, 4)))
    Y = pt.TiledIndexSpace(pt.IndexSpace(trange(0, 3)))

    i, j = labels(X, 2)
    (a,) = labels(Y, 1)

    A = pt.TensorDouble([X, X, Y])
    B = pt.TensorDouble([X, X])

    sch = pt.Scheduler(ec)
    sch.allocate(A, B)
    sch(A(), "=", 3.0)
    sch(B(), "=", 0.0)
    sch(B(i, j), "+=", A(i, j, a))
    sch.execute()

    check_value(A, 3.0)
    check_value(B, 9.0)

    maybe_print_tensor(A, "A tensor")
    maybe_print_tensor(B, "B tensor")

    dealloc(A, B)


def test_slack_issues():
    rprint("Slack Issue Start")

    pg, ec = make_ec()
    sch = pt.Scheduler(ec)

    AOs_ = pt.IndexSpace(trange(0, 10))
    MOs_ = pt.IndexSpace(
        trange(0, 10),
        {
            "O": [trange(0, 5)],
            "V": [trange(5, 10)],
        },
    )

    tAOs = pt.TiledIndexSpace(AOs_)
    tMOs = pt.TiledIndexSpace(MOs_)
    tXYZ = pt.TiledIndexSpace(pt.IndexSpace(trange(0, 3)))

    D = pt.TensorDouble([tXYZ, tAOs, tAOs])
    C = pt.TensorDouble([tAOs, tMOs])
    W = pt.TensorDouble([tMOs("O"), tMOs("O")])

    sch.allocate(C, W, D)
    sch(C(), "=", 42.0)
    sch(W(), "=", 1.0)
    sch(D(), "=", 1.0)
    sch.execute()

    xyz = tXYZ
    AOs = C.tiled_index_spaces()[0]
    MOs = C.tiled_index_spaces()[1]("O")

    initialMO_state = pt.TensorDouble([xyz, MOs, MOs])
    tmp = pt.TensorDouble([xyz, AOs, MOs])

    (x,) = labels(xyz, 1)
    mu, nu = labels(AOs, 2)
    i, j = labels(MOs, 2)

    _tmp_lbls = tmp().labels()
    _D_lbls = D().labels()
    _C_lbls = C().labels()

    sch.allocate(initialMO_state, tmp)
    sch(tmp(x, mu, i), "=", D(x, mu, nu) * C(nu, i))
    sch(initialMO_state(x, i, j), "=", C(mu, i) * tmp(x, mu, j))
    sch.execute()

    X = initialMO_state.tiled_index_spaces()[0]
    n_MOs = W.tiled_index_spaces()[0]
    n_LMOs = W.tiled_index_spaces()[1]

    (x_,) = labels(X, 1)
    r_, s_ = labels(n_MOs, 2, start=0)
    i_, j_ = labels(n_LMOs, 2, start=10)

    initW = pt.TensorDouble([X, n_MOs, n_LMOs])
    WinitW = pt.TensorDouble([X, n_LMOs, n_LMOs])

    sch.allocate(initW, WinitW)
    sch(initW(x_, r_, i_), "=", initialMO_state(x_, r_, s_) * W(s_, i_))
    sch(WinitW(x_, i_, j_), "=", W(r_, i_) * initW(x_, r_, j_))
    sch.deallocate(initialMO_state, tmp, C, W, D, initW, WinitW)
    sch.execute()


def test_slicing_examples():
    AOs = pt.IndexSpace(trange(0, 10))
    MOs = pt.IndexSpace(
        trange(0, 10),
        {
            "O": [trange(0, 5)],
            "V": [trange(5, 10)],
        },
    )

    tAOs = pt.TiledIndexSpace(AOs)
    tMOs = pt.TiledIndexSpace(MOs)

    A = pt.TensorDouble([tMOs])
    B = pt.TensorDouble([tMOs, tMOs])

    pg, ec = make_ec()
    sch = pt.Scheduler(ec)

    sch.allocate(A, B)
    sch(A(), "=", 0.0)
    sch(B(), "=", 4.0)
    sch.execute()

    (i,) = labels(tMOs, 1)
    (j,) = labels(tMOs("O"), 1)
    (k,) = labels(tMOs("V"), 1)

    sch(B(j, j), "=", 42.0)
    sch(B(k, k), "=", 21.0)
    sch(A(i), "=", B(i, i))
    sch.execute()

    check_value(A(j), 42.0)
    check_value(A(k), 21.0)
    check_value(B(j, j), 42.0)
    check_value(B(k, k), 21.0)

    dealloc(A, B)


def test_fill_tensors_using_lambda_functions():
    AOs = pt.IndexSpace(trange(0, 10))
    MOs = pt.IndexSpace(
        trange(0, 10),
        {
            "O": [trange(0, 5)],
            "V": [trange(5, 10)],
        },
    )

    tAOs = pt.TiledIndexSpace(AOs)
    tMOs = pt.TiledIndexSpace(MOs)

    A = pt.TensorDouble([tAOs, tAOs])
    B = pt.TensorDouble([tMOs, tMOs])

    pg, ec = make_ec()
    A.allocate_self(ec)
    B.allocate_self(ec)

    pt.update_tensor(A(), lambda_function)
    check_value(A, 42.0)

    pt.Scheduler(ec)(A(), "=", 0.0).execute()
    check_value(A, 0.0)

    pt.update_tensor(A(), make_l_func(9))

    i = label(tAOs)
    pt.update_tensor(A(i, i), lambda_function)

    maybe_print_tensor(A, "A after update_tensor tests")

    dealloc(A, B)


def test_execution_context_from_tensor():
    AO = pt.TiledIndexSpace(pt.IndexSpace(trange(10)), 2)

    T0 = pt.TensorDouble([AO, AO])
    T1 = pt.TensorDouble([AO, AO])

    pg, ec = make_ec()

    T0.allocate_self(ec)

    t0_ec = T0.execution_context()
    assert t0_ec is not None, "Allocated tensor returned null execution_context()"

    t1_ec = T1.execution_context()
    assert t1_ec is None, "Unallocated tensor should have null execution_context()"

    T0.deallocate()


def test_apply_ewise():
    MO = pt.IndexSpace(
        trange(10),
        {
            "occ": [trange(0, 5)],
            "virt": [trange(5, 10)],
        },
    )
    tMO = pt.TiledIndexSpace(MO)

    i, j = labels(tMO, 2)
    i_virt, j_virt = labels(tMO("virt"), 2)

    T = pt.TensorDouble([i, j])

    pg, ec = make_ec()
    sch = pt.Scheduler(ec)

    sch.allocate(T)
    sch(T(), "=", 42.0)
    sch(T(i_virt, j_virt), "=", 21.0)
    sch.execute()

    maybe_print_tensor(T, "Printing tensor T")

    Temp = pt.scale(T(i_virt, j_virt), 0.1)

    maybe_print_tensor(Temp, "Printing tensor Temp")
    check_value(Temp, 2.1)

    dealloc(T, Temp)


def test_fill_sparse_tensor():
    AO = pt.TiledIndexSpace(pt.IndexSpace(trange(7)))
    MO = pt.TiledIndexSpace(pt.IndexSpace(trange(10)))

    depMO_1 = {
        (0,): pt.TiledIndexSpace(MO, [1, 4, 5]),
        (2,): pt.TiledIndexSpace(MO, [0, 3, 6, 8]),
        (5,): pt.TiledIndexSpace(MO, [2, 4, 6, 9]),
    }

    MO_AO_1 = pt.TiledIndexSpace(MO, [AO], depMO_1)

    i, j = labels(AO, 2)
    mu, nu = labels(MO, 2)
    mu_i, nu_i = labels(MO_AO_1, 2)

    pg, ec = make_ec()
    sch = pt.Scheduler(ec)

    Q = pt.TensorDouble([i, mu_i(i)])
    P = pt.TensorDouble([i, mu_i(i)])
    T = pt.TensorDouble([i, mu])

    sch.allocate(Q, P, T)
    sch(Q(), "=", 1.0)
    sch(P(), "=", 2.0)
    sch(T(), "=", 3.0)
    sch.execute()

    maybe_print_tensor(Q, "Q Tensor")
    maybe_print_tensor(P, "P Tensor")
    maybe_print_tensor(T, "T Tensor")

    pt.fill_sparse_tensor_double(Q, lambda_function)

    maybe_print_tensor(Q, "Q Tensor after fill_sparse_tensor")
    check_value(Q, 42.0)

    dealloc(Q, P, T)


TESTS = [
    ("Block Sparse Tensor Construction", test_block_sparse_tensor_construction),
    ("Spin Tensor Construction", test_spin_tensor_construction),
    ("Hash Based Equality and Compatibility Check", test_hash_based_equality_and_compatibility_check),
    ("GitHub Issues", test_github_issues),
    ("Slack Issues", test_slack_issues),
    ("Slicing examples", test_slicing_examples),
    ("Fill tensors using lambda functions", test_fill_tensors_using_lambda_functions),
    ("Test case for getting ExecutionContext from a Tensor", test_execution_context_from_tensor),
    ("Test for apply_ewise", test_apply_ewise),
    ("Testing fill_sparse_tensor", test_fill_sparse_tensor),
]


def main():
    # Avoid passing script-only options such as --verbose to TAMM.
    pt.initialize([sys.argv[0]], False)

    try:
        for name, fn in TESTS:
            rprint(f"[ RUN      ] {name}")
            fn()
            rprint(f"[       OK ] {name}")
    finally:
        pt.finalize(True)


if __name__ == "__main__":
    main()
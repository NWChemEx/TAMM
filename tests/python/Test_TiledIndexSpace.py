import pytamm as pt


def tile_values(tis, tile_id):
    """
    The current Python binding exposes block_begin(i) and block_end(i) as
    first and last index values, not iterators. So this reconstructs the
    contiguous tile values checked in the C++ tests.
    """
    b = int(tis.block_begin(tile_id))
    e = int(tis.block_end(tile_id))
    return list(range(b, e + 1))


def check_tiles(tis, tiled_iv):
    assert int(tis.num_tiles()) == len(tiled_iv)
    for i, ref in enumerate(tiled_iv):
        got = tile_values(tis, i)
        assert got == ref, f"tile {i}: got {got}, expected {ref}"


def print_dependency(tis, name=""):
    """
    Requires optional bindings:
      TiledIndexSpace.tiled_dep_map()
      TiledIndexSpace.ref_indices()

    If unavailable, this just reports that dependency printing is skipped.
    """
    if name:
        print(name)

    if not hasattr(tis, "tiled_dep_map"):
        print("Dependency Map: <tiled_dep_map not bound>")
        return

    dep_map = tis.tiled_dep_map()
    print("Dependency Map")

    for key, subtis in dep_map.items():
        try:
            ref = list(subtis.ref_indices())
        except AttributeError:
            ref = "<ref_indices not bound>"
        print(f"{tuple(key)} -> {ref}")


def test_tiled_index_space_construction():
    is_ = pt.IndexSpace(
        tuple(range(10, 20)),
        {
            "occ": [pt.range(0, 5)],
            "virt": [pt.range(5, 10)],
            "alpha": [pt.range(0, 3), pt.range(5, 8)],
            "beta": [pt.range(3, 5), pt.range(8, 10)],
        },
    )

    t5_is = pt.TiledIndexSpace(is_, 5)
    t3_is = pt.TiledIndexSpace(is_, 3)

    assert t5_is.index_space() == is_
    assert t3_is.index_space() == is_

    i, j = t5_is.labels("all", count=2)

    assert i.tiled_index_space() == t5_is
    assert j.tiled_index_space() == t5_is

    k = t3_is.label("all", 1)

    assert k.tiled_index_space() == t3_is


def test_tiled_index_space_construction_with_multiple_tile_size():
    is1 = pt.IndexSpace(pt.range(10, 20))
    tis1 = pt.TiledIndexSpace(is1, [2, 3, 5])

    tiled_iv = [
        [10, 11],
        [12, 13, 14],
        [15, 16, 17, 18, 19],
    ]

    check_tiles(tis1, tiled_iv)


def test_tiled_index_space_construction_with_multiple_tile_size_named_subspaces():
    is1 = pt.IndexSpace(
        pt.range(10, 20),
        {
            "occ": [pt.range(0, 5)],
            "virt": [pt.range(5, 10)],
        },
    )

    tis1 = pt.TiledIndexSpace(is1, [2, 3, 5])

    tiled_iv = [
        [10, 11],
        [12, 13, 14],
        [15, 16, 17, 18, 19],
    ]

    check_tiles(tis1, tiled_iv)


def test_tiled_index_space_tiling_check():
    tempIS1 = pt.IndexSpace(
        pt.range(10, 50),
        {
            "occ": [pt.range(0, 20)],
            "virt": [pt.range(20, 40)],
            "alpha": [pt.range(0, 13), pt.range(20, 33)],
            "beta": [pt.range(13, 20), pt.range(33, 40)],
        },
    )

    tempIS2 = pt.IndexSpace(
        pt.range(50, 90),
        {
            "occ": [pt.range(0, 20)],
            "virt": [pt.range(20, 40)],
            "alpha": [pt.range(0, 8), pt.range(20, 28)],
            "beta": [pt.range(8, 20), pt.range(28, 40)],
        },
    )

    tempIS3 = pt.IndexSpace(
        [tempIS1, tempIS2],
        ["occ", "virt"],
        {},
        {
            "alpha": ["occ:alpha", "virt::alpha"],
            "beta": ["occ:beta", "virt:beta"],
        },
    )

    t10_is = pt.TiledIndexSpace(tempIS3, 10)

    tiled_iv = [
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21, 22],
        [23, 24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42],
        [43, 44, 45, 46, 47, 48, 49],
        [50, 51, 52, 53, 54, 55, 56, 57],
        [58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        [68, 69],
        [70, 71, 72, 73, 74, 75, 76, 77],
        [78, 79, 80, 81, 82, 83, 84, 85, 86, 87],
        [88, 89],
    ]

    check_tiles(t10_is, tiled_iv)


def test_tiled_index_space_tiling_with_different_named_subspaces():
    tempIS1 = pt.IndexSpace(
        pt.range(10, 50),
        {
            "occ": [pt.range(0, 20)],
            "virt": [pt.range(20, 40)],
            "alpha": [pt.range(0, 12), pt.range(20, 33)],
            "beta": [pt.range(13, 20), pt.range(33, 40)],
        },
    )

    tempIS2 = pt.IndexSpace(
        pt.range(50, 90),
        {
            "occ": [pt.range(0, 20)],
            "virt": [pt.range(20, 40)],
            "alpha": [pt.range(0, 8), pt.range(20, 28)],
            "beta": [pt.range(8, 20), pt.range(28, 40)],
        },
    )

    tempIS3 = pt.IndexSpace(
        [tempIS1, tempIS2],
        ["occ", "virt"],
        {},
        {
            "alpha": ["occ:alpha", "virt::alpha"],
            "beta": ["occ:beta", "virt:beta"],
        },
    )

    t10_is = pt.TiledIndexSpace(tempIS3, 10)

    tiled_iv = [
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [20, 21],
        [22],
        [23, 24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42],
        [43, 44, 45, 46, 47, 48, 49],
        [50, 51, 52, 53, 54, 55, 56, 57],
        [58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
        [68, 69],
        [70, 71, 72, 73, 74, 75, 76, 77],
        [78, 79, 80, 81, 82, 83, 84, 85, 86, 87],
        [88, 89],
    ]

    check_tiles(t10_is, tiled_iv)


def expect_no_throw(fn):
    try:
        fn()
        return True
    except Exception:
        return False


def expect_throw(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def test_tiled_index_space_construction_checks():
    simple_is = pt.IndexSpace(pt.range(10, 20))
    named_is = pt.IndexSpace(
        pt.range(10, 20),
        {
            "occ": [pt.range(0, 5)],
            "virt": [pt.range(5, 10)],
        },
    )

    assert expect_no_throw(lambda: pt.TiledIndexSpace(simple_is))
    assert expect_no_throw(lambda: pt.TiledIndexSpace(named_is))
    assert expect_no_throw(lambda: pt.TiledIndexSpace(simple_is, 5))
    assert expect_no_throw(lambda: pt.TiledIndexSpace(named_is, 5))
    assert expect_no_throw(lambda: pt.TiledIndexSpace(simple_is, 20))
    assert expect_no_throw(lambda: pt.TiledIndexSpace(named_is, 20))

    tis_full = pt.TiledIndexSpace(simple_is)
    tis_named = pt.TiledIndexSpace(named_is)

    assert expect_no_throw(lambda: pt.TiledIndexSpace(tis_full, [0]))
    assert expect_no_throw(lambda: pt.TiledIndexSpace(tis_full, [0, 1]))
    assert expect_throw(lambda: pt.TiledIndexSpace(tis_full, [100]))

    assert expect_no_throw(lambda: pt.TiledIndexSpace(tis_full, pt.range(1)))
    assert expect_no_throw(lambda: pt.TiledIndexSpace(tis_full, pt.range(0, 2)))
    assert expect_throw(lambda: pt.TiledIndexSpace(tis_full, pt.range(100)))

    N = tis_named("all")
    O = tis_named("occ")
    V = tis_named("virt")

    assert O.is_compatible_with(N)
    assert V.is_compatible_with(N)


def test_tiled_index_space_construction_with_spin_attributes():
    is_att = pt.IndexSpace(
        pt.range(0, 20),
        {
            "occ": [pt.range(0, 10)],
            "virt": [pt.range(10, 20)],
        },
        {
            pt.Spin(2): [pt.range(0, 5), pt.range(10, 15)],
            pt.Spin(1): [pt.range(5, 10), pt.range(15, 20)],
        },
    )

    assert is_att.has_spin() is True

    tilesizes = [3, 2, 3, 2, 3, 2, 3, 2]

    tis_att = pt.TiledIndexSpace(is_att, 3)
    tis_att2 = pt.TiledIndexSpace(is_att, tilesizes)

    assert int(tis_att.num_tiles()) == 8
    assert int(tis_att2.num_tiles()) == 8

    for i in range(int(tis_att.num_tiles())):
        assert int(tis_att.tile_size(i)) == tilesizes[i]
        assert int(tis_att2.tile_size(i)) == tilesizes[i]


def test_tiled_index_space_operations():
    AOs = pt.TiledIndexSpace(pt.IndexSpace(pt.range(7)))
    MOs = pt.TiledIndexSpace(pt.IndexSpace(pt.range(4)))

    new_dep = {
        (0, 0): pt.TiledIndexSpace(AOs, [0, 3, 4]),
        (1, 5): pt.TiledIndexSpace(AOs, [0, 3, 6]),
        (2, 0): pt.TiledIndexSpace(AOs, [1, 3, 5]),
        (2, 5): pt.TiledIndexSpace(AOs, [0, 5]),
    }

    dep_nu_mu_q = {
        (0,): pt.TiledIndexSpace(AOs, [0, 3, 4]),
        (1,): pt.TiledIndexSpace(AOs, [0, 3, 6]),
        (2,): pt.TiledIndexSpace(AOs, [1, 3, 5]),
    }

    dep_nu_mu_d = {
        (0,): pt.TiledIndexSpace(AOs, [1, 3, 5]),
        (1,): pt.TiledIndexSpace(AOs, [0, 1, 2]),
        (2,): pt.TiledIndexSpace(AOs, [0, 2, 4]),
        (3,): pt.TiledIndexSpace(AOs, [1, 6]),
        (4,): pt.TiledIndexSpace(AOs, [3, 5]),
        (6,): pt.TiledIndexSpace(AOs, [0, 1, 2]),
    }

    dep_nu_mu_c = {
        (0,): pt.TiledIndexSpace(AOs, [3]),
        (2,): pt.TiledIndexSpace(AOs, [0, 2]),
        (3,): pt.TiledIndexSpace(AOs, [1]),
        (4,): pt.TiledIndexSpace(AOs, [3, 5]),
        (6,): pt.TiledIndexSpace(AOs, [2]),
    }

    test_tis = pt.TiledIndexSpace(AOs, [MOs, AOs], new_dep)
    tSubAO_AO_Q = pt.TiledIndexSpace(AOs, [MOs], dep_nu_mu_q)
    tSubAO_AO_D = pt.TiledIndexSpace(AOs, [AOs], dep_nu_mu_d)
    tSubAO_AO_C = pt.TiledIndexSpace(AOs, [AOs], dep_nu_mu_c)

    print_dependency(tSubAO_AO_Q, "tSubAO_AO_Q")

    inv_tSubAO_AO_Q = pt.invert_tis(tSubAO_AO_Q)
    print_dependency(inv_tSubAO_AO_Q, "inv_tSubAO_AO_Q")

    comp_tSubAO_AO_Q_D = pt.compose_tis(tSubAO_AO_Q, tSubAO_AO_D)
    print_dependency(tSubAO_AO_D, "tSubAO_AO_D")
    print_dependency(comp_tSubAO_AO_Q_D, "comp_tSubAO_AO_Q_D")

    union_tSubAO_AO_D_C = pt.union_tis(tSubAO_AO_D, tSubAO_AO_C)
    print_dependency(tSubAO_AO_D, "tSubAO_AO_D")
    print_dependency(tSubAO_AO_C, "tSubAO_AO_C")
    print_dependency(union_tSubAO_AO_D_C, "union_tSubAO_AO_D_C")

    project_test_tis = pt.project_tis(test_tis, MOs)
    print_dependency(test_tis, "test_tis")
    print_dependency(project_test_tis, "project_test_tis")

    project_MO_Q = pt.project_tis(tSubAO_AO_Q, MOs)
    print("project_MO_Q")
    if hasattr(project_MO_Q, "ref_indices"):
        print(list(project_MO_Q.ref_indices()))
    else:
        print("<ref_indices not bound>")

    tis_1 = pt.TiledIndexSpace(AOs, [1, 2, 5])
    tis_2 = pt.TiledIndexSpace(AOs, [2, 3, 6])

    u_tis12 = pt.union_tis(tis_1, tis_2)

    print("u_tis12")
    if hasattr(u_tis12, "ref_indices"):
        print(list(u_tis12.ref_indices()))
    else:
        print("<ref_indices not bound>")


def main():
    pt.initialize(["Test_TiledIndexSpace.py"], False)

    try:
        test_tiled_index_space_construction()
        test_tiled_index_space_construction_with_multiple_tile_size()
        test_tiled_index_space_construction_with_multiple_tile_size_named_subspaces()
        test_tiled_index_space_tiling_check()
        test_tiled_index_space_tiling_with_different_named_subspaces()
        test_tiled_index_space_construction_checks()
        test_tiled_index_space_construction_with_spin_attributes()
        test_tiled_index_space_operations()

    finally:
        pt.finalize()


if __name__ == "__main__":
    main()
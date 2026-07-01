import pytamm as pt


def as_tuple(x):
    return tuple(int(v) for v in x)


def check_indices(index_space, expected):
    expected = tuple(expected)
    assert index_space.num_indices() == len(expected)
    assert as_tuple(index_space) == expected


def check_spin_attributes(index_space, indices, spin_value):
    for idx in indices:
        assert int(index_space.spin(idx).value()) == int(spin_value.value())


def check_index_space_equality():
    is1 = pt.IndexSpace(pt.range(10))
    is2 = pt.IndexSpace(pt.range(0, 10))
    is3 = pt.IndexSpace(pt.range(0, 20))
    is4 = pt.IndexSpace(pt.range(0, 20, 2))
    is5 = pt.IndexSpace(
        pt.range(10),
        {
            "occ": [pt.range(0, 5)],
            "virt": [pt.range(5, 10)],
        },
    )
    is6 = pt.IndexSpace(is3, pt.range(10))
    is7 = pt.IndexSpace([is5("occ"), is5("virt")])

    assert is1 == is1
    assert is1 == is2
    assert is1 != is3
    assert is1 != is4
    assert is3 != is4
    assert is1 == is5
    assert is1 == is6
    assert is1 == is7
    assert is2 == is5
    assert is2 == is6
    assert is2 == is7
    assert is5 == is5
    assert is5 == is6
    assert is5 == is7
    assert is6 == is7


def test_index_space_construction_with_ranges():
    is0 = pt.IndexSpace(pt.range(10))
    iv0 = tuple(range(10))
    check_indices(is0, iv0)
    check_indices(is0("all"), iv0)

    is1 = pt.IndexSpace(pt.range(10, 20))
    iv1 = tuple(range(10, 20))
    check_indices(is1, iv1)
    check_indices(is1("all"), iv1)

    is2 = pt.IndexSpace(pt.range(10, 20, 2))
    iv2 = (10, 12, 14, 16, 18)
    check_indices(is2, iv2)
    check_indices(is2("all"), iv2)


def test_index_space_construction_with_set_of_indices():
    is1 = pt.IndexSpace((10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
    iv1 = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
    check_indices(is1, iv1)
    check_indices(is1("all"), iv1)


def test_index_space_construction_with_named_subspaces():
    is2 = pt.IndexSpace(
        (10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        {
            "occ": [pt.range(0, 5)],
            "virt": [pt.range(5, 10)],
            "alpha": [pt.range(0, 3), pt.range(5, 8)],
            "beta": [pt.range(3, 5), pt.range(8, 10)],
        },
    )

    check_indices(is2, (10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
    check_indices(is2("all"), (10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
    check_indices(is2("occ"), (10, 11, 12, 13, 14))
    check_indices(is2("virt"), (15, 16, 17, 18, 19))
    check_indices(is2("alpha"), (10, 11, 12, 15, 16, 17))
    check_indices(is2("beta"), (13, 14, 18, 19))


def test_index_space_construction_by_range_with_named_subspaces():
    is0 = pt.IndexSpace(
        pt.range(10),
        {
            "occ": [pt.range(2, 5)],
            "virt": [pt.range(4, 9)],
        },
    )

    check_indices(is0, tuple(range(10)))
    check_indices(is0("all"), tuple(range(10)))
    check_indices(is0("occ"), (2, 3, 4))
    check_indices(is0("virt"), (4, 5, 6, 7, 8))


def test_retrieval_of_point_in_index_space():
    is0 = pt.IndexSpace((12, 11, 14, 24, 9, 8, 7))

    assert int(is0.index(3)) == 24
    assert int(is0[5]) == 8


def test_index_space_construction_by_concatenation():
    is1 = pt.IndexSpace((2, 4, 5, 6, 7, 8))
    is2 = pt.IndexSpace((3, 7, 9))
    is0 = pt.IndexSpace([is1, is2])

    check_indices(is0, (2, 4, 5, 6, 7, 8, 3, 7, 9))


def test_index_space_construction_with_named_subspaces_by_concatenation():
    is1 = pt.IndexSpace((2, 4, 5))
    is2 = pt.IndexSpace((1, 3))
    is3 = pt.IndexSpace((3, 6))

    is0 = pt.IndexSpace(
        [is1, is2, is3],
        ["temp1", "temp2", "temp3"],
        {
            "occ": [pt.range(2, 3)],
            "virt": [pt.range(1, 4)],
            "alpha": [pt.range(0, 7, 2)],
            "beta": [pt.range(1, 7, 2)],
        },
    )

    check_indices(is0, (2, 4, 5, 1, 3, 3, 6))
    check_indices(is0("all"), (2, 4, 5, 1, 3, 3, 6))
    check_indices(is0("occ"), (5,))
    check_indices(is0("virt"), (4, 5, 1))
    check_indices(is0("alpha"), (2, 5, 3, 6))
    check_indices(is0("beta"), (4, 1, 3))


def test_index_space_construction_by_subspacing():
    is0 = pt.IndexSpace((9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
    sub_is = pt.IndexSpace(is0, pt.range(2, 8))

    check_indices(sub_is, (7, 6, 5, 4, 3, 2))
    check_indices(sub_is("all"), (7, 6, 5, 4, 3, 2))


def test_index_space_construction_by_subspacing_with_named_subspaces():
    is0 = pt.IndexSpace((9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
    sub_is = pt.IndexSpace(
        is0,
        pt.range(2, 8),
        {
            "occ": [pt.range(0, 3)],
            "virt": [pt.range(3, 6)],
        },
    )

    check_indices(sub_is, (7, 6, 5, 4, 3, 2))
    check_indices(sub_is("all"), (7, 6, 5, 4, 3, 2))
    check_indices(sub_is("occ"), (7, 6, 5))
    check_indices(sub_is("virt"), (4, 3, 2))


def test_index_space_construction_by_aggregating_with_other_index_spaces():
    temp_is1 = pt.IndexSpace((10, 12, 14, 16, 18))
    temp_is2 = pt.IndexSpace((1, 3, 5, 7, 9))
    agg_is = pt.IndexSpace([temp_is1, temp_is2])

    check_indices(agg_is, (10, 12, 14, 16, 18, 1, 3, 5, 7, 9))
    check_indices(agg_is("all"), (10, 12, 14, 16, 18, 1, 3, 5, 7, 9))


def test_index_space_construction_by_aggregating_with_subnames():
    temp_is = pt.IndexSpace(
        (10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
        {
            "occ": [pt.range(0, 5)],
            "virt": [pt.range(5, 10)],
            "alpha": [pt.range(0, 3), pt.range(5, 8)],
            "beta": [pt.range(3, 5), pt.range(8, 10)],
        },
    )
    temp_is1 = pt.IndexSpace(
        pt.range(20, 30),
        {
            "occ": [pt.range(0, 5)],
            "virt": [pt.range(5, 10)],
            "alpha": [pt.range(0, 3), pt.range(5, 8)],
            "beta": [pt.range(3, 5), pt.range(8, 10)],
        },
    )

    agg_is = pt.IndexSpace(
        [temp_is, temp_is1],
        ["occ", "virt"],
        {
            "local": [pt.range(8, 13)],
        },
        {
            "alpha": ["occ:alpha", "virt:alpha"],
            "beta": ["occ:beta", "virt:beta"],
        },
    )

    full_indices = pt.construct_index_vector(pt.range(10, 30))

    check_indices(agg_is, full_indices)
    check_indices(agg_is("all"), full_indices)
    check_indices(agg_is("occ"), pt.construct_index_vector(pt.range(10, 20)))
    check_indices(agg_is("virt"), pt.construct_index_vector(pt.range(20, 30)))
    check_indices(agg_is("local"), (18, 19, 20, 21, 22))
    check_indices(agg_is("alpha"), (10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27))
    check_indices(agg_is("beta"), (13, 14, 18, 19, 23, 24, 28, 29))


def test_index_space_construction_using_dependent_index_spaces():
    is1 = pt.IndexSpace(pt.range(10, 20))
    is2 = pt.IndexSpace(pt.range(10, 20, 2))
    temp_is1 = pt.IndexSpace((0, 1, 2))
    temp_is2 = pt.IndexSpace((8, 9))

    t_is1 = pt.TiledIndexSpace(temp_is1, 2)
    t_is2 = pt.TiledIndexSpace(temp_is2, 1)

    dep_relation = {
        (0, 0): is1,
        (0, 1): is2,
        (1, 0): temp_is2,
        (1, 1): is2,
    }

    dep_is = pt.IndexSpace([t_is1, t_is2], dep_relation)

    assert dep_is((0, 0)) == is1
    assert dep_is((0, 1)) == is2
    assert dep_is((1, 0)) == temp_is2
    assert dep_is((1, 1)) == is2

def test_index_space_construction_for_sub_ao_space_dependent_over_atom_index_space():
    AO = pt.IndexSpace(pt.range(0, 20))
    ATOM = pt.IndexSpace((0, 1, 2, 3, 4))
    T_ATOM = pt.TiledIndexSpace(ATOM, 2)

    ao_atom_relation = {
        (0,): pt.IndexSpace(AO, (3, 4, 7)),
        (1,): pt.IndexSpace(AO, (1, 5, 7)),
        (2,): pt.IndexSpace(AO, (1, 9, 11)),
        (3,): pt.IndexSpace(AO, (11, 14)),
        (4,): pt.IndexSpace(AO, (2, 5, 13, 17)),
    }

    pt.IndexSpace([T_ATOM], AO, ao_atom_relation)

def test_index_space_construction_with_spin_for_all_subspaces():
    is0 = pt.IndexSpace(
        pt.range(100),
        {
            "occ": [pt.range(0, 50)],
            "virt": [pt.range(50, 100)],
            "alpha": [pt.range(0, 25), pt.range(50, 75)],
            "beta": [pt.range(25, 50), pt.range(75, 100)],
        },
        {
            pt.Spin(1): [pt.range(0, 25), pt.range(50, 75)],
            pt.Spin(2): [pt.range(25, 50), pt.range(75, 100)],
        },
    )

    full_indices = pt.construct_index_vector(pt.range(100))
    alpha_indices = pt.construct_index_vector([pt.range(0, 25), pt.range(50, 75)])
    beta_indices = pt.construct_index_vector([pt.range(25, 50), pt.range(75, 100)])

    check_indices(is0, full_indices)
    check_indices(is0("all"), full_indices)
    check_indices(is0("occ"), pt.construct_index_vector(pt.range(0, 50)))
    check_indices(is0("virt"), pt.construct_index_vector(pt.range(50, 100)))
    check_indices(is0("alpha"), alpha_indices)
    check_indices(is0("beta"), beta_indices)

    check_spin_attributes(is0, alpha_indices, pt.Spin(1))
    check_spin_attributes(is0, beta_indices, pt.Spin(2))
    check_spin_attributes(is0("occ"), pt.construct_index_vector(pt.range(0, 25)), pt.Spin(1))
    check_spin_attributes(is0("occ"), pt.construct_index_vector(pt.range(25, 50)), pt.Spin(2))
    check_spin_attributes(is0("virt"), pt.construct_index_vector(pt.range(0, 25)), pt.Spin(1))
    check_spin_attributes(is0("virt"), pt.construct_index_vector(pt.range(25, 50)), pt.Spin(2))


def main():
    tests = [
        check_index_space_equality,
        test_index_space_construction_with_ranges,
        test_index_space_construction_with_set_of_indices,
        test_index_space_construction_with_named_subspaces,
        test_index_space_construction_by_range_with_named_subspaces,
        test_retrieval_of_point_in_index_space,
        test_index_space_construction_by_concatenation,
        test_index_space_construction_with_named_subspaces_by_concatenation,
        test_index_space_construction_by_subspacing,
        test_index_space_construction_by_subspacing_with_named_subspaces,
        test_index_space_construction_by_aggregating_with_other_index_spaces,
        test_index_space_construction_by_aggregating_with_subnames,
        test_index_space_construction_using_dependent_index_spaces,
        test_index_space_construction_for_sub_ao_space_dependent_over_atom_index_space,
        test_index_space_construction_with_spin_for_all_subspaces,
    ]

    for test in tests:
        test()


if __name__ == "__main__":
    main()
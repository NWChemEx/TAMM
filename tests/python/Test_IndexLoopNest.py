import pytamm as pt


def assert_loop_values(iln, expected):
    got = [tuple(int(x) for x in v) for v in iln]
    expected = [tuple(int(x) for x in v) for v in expected]
    assert got == expected


def test_zero_dimensional_index_loop_nest_with_index_bound_constructor():
    iln = pt.IndexLoopNest()
    values = list(iln)
    assert len(values) == 1


def test_zero_dimensional_index_loop_nest_with_list_of_arguments_constructor():
    iln = pt.IndexLoopNest([])
    values = list(iln)
    assert len(values) == 1


def test_one_dimensional_index_loop_nest_with_index_bound_constructor():
    is0 = pt.IndexSpace(pt.range(10))
    tis = pt.TiledIndexSpace(is0, 1)
    i, = tis.labels("all", count=1)

    iln = pt.IndexLoopNest(i)

    expected = [(cnt,) for cnt in range(10)]
    assert_loop_values(iln, expected)


def test_one_dimensional_index_loop_nest_with_list_of_arguments_constructor():
    is0 = pt.IndexSpace(pt.range(10))
    tis = pt.TiledIndexSpace(is0, 1)

    iln = pt.IndexLoopNest([tis])

    expected = [(cnt,) for cnt in range(10)]
    assert_loop_values(iln, expected)


def test_two_dimensional_square_index_loop_nest_with_index_bound_constructor():
    is0 = pt.IndexSpace(pt.range(10))
    tis = pt.TiledIndexSpace(is0, 1)
    i, j = tis.labels("all", count=2)

    iln = pt.IndexLoopNest([i, j])

    expected = [(ci, cj) for ci in range(10) for cj in range(10)]
    assert_loop_values(iln, expected)


def test_two_dimensional_square_index_loop_nest_with_list_of_arguments_constructor():
    is0 = pt.IndexSpace(pt.range(10))
    tis = pt.TiledIndexSpace(is0, 1)

    iln = pt.IndexLoopNest([tis, tis])

    expected = [(ci, cj) for ci in range(10) for cj in range(10)]
    assert_loop_values(iln, expected)


def test_two_dimensional_rectangular_index_loop_nest():
    ri1 = 9
    ri2 = 23

    is1 = pt.IndexSpace(pt.range(ri1))
    is2 = pt.IndexSpace(pt.range(ri2))

    tis1 = pt.TiledIndexSpace(is1, 1)
    tis2 = pt.TiledIndexSpace(is2, 1)

    til1, = tis1.labels("all", count=1)
    til2, = tis2.labels("all", count=1)

    iln = pt.IndexLoopNest([til1, til2])

    expected = [(c1, c2) for c1 in range(ri1) for c2 in range(ri2)]
    assert_loop_values(iln, expected)


def test_two_dimensional_upper_triangular_index_loop_nest():
    ri = 11

    is0 = pt.IndexSpace(pt.range(ri))
    tis = pt.TiledIndexSpace(is0, 1)

    i, j = tis.labels("all", count=2)

    bi = pt.IndexLoopBound(i)
    bj = pt.IndexLoopBound(j)

    iln = pt.IndexLoopNest([bi, bj + (bj >= bi)])

    expected = [(ci, cj) for ci in range(ri) for cj in range(ci, ri)]
    assert_loop_values(iln, expected)


def test_two_dimensional_lower_triangular_index_loop_nest():
    ri = 11

    is0 = pt.IndexSpace(pt.range(ri))
    tis = pt.TiledIndexSpace(is0, 1)

    i, j = tis.labels("all", count=2)

    bi = pt.IndexLoopBound(i)
    bj = pt.IndexLoopBound(j)

    iln = pt.IndexLoopNest([bi, bj + (bj <= bi)])

    expected = [(ci, cj) for ci in range(ri) for cj in range(ci + 1)]
    assert_loop_values(iln, expected)


def test_two_dimensional_diagonal_index_loop_nest():
    ri = 11

    is0 = pt.IndexSpace(pt.range(ri))
    tis = pt.TiledIndexSpace(is0, 1)

    i, j = tis.labels("all", count=2)

    bi = pt.IndexLoopBound(i)
    bj = pt.IndexLoopBound(j)

    iln = pt.IndexLoopNest([bi, bj + (bj <= bi) + (bj >= bi)])

    expected = [(ci, ci) for ci in range(ri)]
    assert_loop_values(iln, expected)


def test_three_dimensional_diagonal_index_loop_nest():
    ri = 11

    is0 = pt.IndexSpace(pt.range(ri))
    tis = pt.TiledIndexSpace(is0, 1)

    i, j, k = tis.labels("all", count=3)

    bi = pt.IndexLoopBound(i)
    bj = pt.IndexLoopBound(j)
    bk = pt.IndexLoopBound(k)

    iln = pt.IndexLoopNest([bi, bj + (bj <= bi) + (bj >= bi), bk == bj])

    expected = [(ci, ci, ci) for ci in range(ri)]
    assert_loop_values(iln, expected)


def test_one_dimensional_split_index_loop_nest_with_index_bound_constructor():
    is0 = pt.IndexSpace(
        pt.range(20),
        {
            "r1": [pt.range(0, 10)],
            "r2": [pt.range(10, 20)],
        },
    )

    tis = pt.TiledIndexSpace(is0, 1)
    i, = tis.labels("all", count=1)

    iln = pt.IndexLoopNest(i)

    expected = [(cnt,) for cnt in range(20)]
    assert_loop_values(iln, expected)


def test_two_dimensional_split_index_loop_nest_with_index_bound_constructor():
    is0 = pt.IndexSpace(
        pt.range(20),
        {
            "r1": [pt.range(0, 10)],
            "r2": [pt.range(10, 20)],
        },
    )

    tis = pt.TiledIndexSpace(is0, 10)
    i, j = tis.labels("all", count=2)

    iln = pt.IndexLoopNest([i, j])

    expected = [(c1, c2) for c1 in range(2) for c2 in range(2)]
    assert_loop_values(iln, expected)


def test_three_dimensional_split_index_loop_nest_with_index_bound_constructor():
    tilesize = 10

    is0 = pt.IndexSpace(
        pt.range(20),
        {
            "r1": [pt.range(0, 10)],
            "r2": [pt.range(10, 20)],
        },
    )

    tis = pt.TiledIndexSpace(is0, tilesize)
    i, j, k = tis.labels("all", count=3)

    iln = pt.IndexLoopNest([i, j, k])

    expected = [
        (c1, c2, c3)
        for c1 in range(20 // tilesize)
        for c2 in range(20 // tilesize)
        for c3 in range(20 // tilesize)
    ]

    assert_loop_values(iln, expected)


def test_four_dimensional_split_index_loop_nest_with_index_bound_constructor():
    tilesize = 10

    is0 = pt.IndexSpace(
        pt.range(20),
        {
            "r1": [pt.range(0, 10)],
            "r2": [pt.range(10, 20)],
        },
    )

    tis = pt.TiledIndexSpace(is0, tilesize)
    i, j, k, l = tis.labels("all", count=4)

    iln = pt.IndexLoopNest([i, j, k, l])

    expected = [
        (c1, c2, c3, c4)
        for c1 in range(20 // tilesize)
        for c2 in range(20 // tilesize)
        for c3 in range(20 // tilesize)
        for c4 in range(20 // tilesize)
    ]

    assert_loop_values(iln, expected)


def main():
    tests = [
        test_zero_dimensional_index_loop_nest_with_index_bound_constructor,
        test_zero_dimensional_index_loop_nest_with_list_of_arguments_constructor,
        test_one_dimensional_index_loop_nest_with_index_bound_constructor,
        test_one_dimensional_index_loop_nest_with_list_of_arguments_constructor,
        test_two_dimensional_square_index_loop_nest_with_index_bound_constructor,
        test_two_dimensional_square_index_loop_nest_with_list_of_arguments_constructor,
        test_two_dimensional_rectangular_index_loop_nest,
        test_two_dimensional_upper_triangular_index_loop_nest,
        test_two_dimensional_lower_triangular_index_loop_nest,
        test_two_dimensional_diagonal_index_loop_nest,
        test_three_dimensional_diagonal_index_loop_nest,
        test_one_dimensional_split_index_loop_nest_with_index_bound_constructor,
        test_two_dimensional_split_index_loop_nest_with_index_bound_constructor,
        test_three_dimensional_split_index_loop_nest_with_index_bound_constructor,
        test_four_dimensional_split_index_loop_nest_with_index_bound_constructor,
    ]

    for test in tests:
        test()


if __name__ == "__main__":
    main()
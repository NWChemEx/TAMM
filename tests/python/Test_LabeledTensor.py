# test_zero_dim_tensor.py
#
# pytest translation of the doctest/C++ test.
#
# Note:
# The Python bindings wrap labeled tensors in PyLabeledTensor and do not always
# force C++ Tensor::operator() validation immediately for label-count cases.
# checked_lt() forces materialization through a harmless OpExpr construction so
# these tests match the C++ test intent.

import pytamm as tamm


def failed(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def checked_lt(tensor, *args):
    lt = tensor(*args)

    # Force C++ labeled-tensor validation without executing anything.
    # This only constructs an OpExpr.
    tamm.copy(lt, lt)

    return lt


def make_tis(n=10):
    is_ = tamm.IndexSpace(tamm.range(n))
    return tamm.TiledIndexSpace(is_)


def tis_labels(tis, count):
    return tis.labels("all", None, count)


def test_zero_dimensional_tensor():
    T1 = tamm.TensorDouble()

    assert not failed(lambda: checked_lt(T1))


def test_zero_dimensional_tensor_wrong_str_count():
    T1 = tamm.TensorDouble()

    assert failed(lambda: checked_lt(T1, "a"))


def test_zero_dimensional_tensor_wrong_label_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble()

    # C++ uses T1(tis), relying on C++ conversion/overload behavior.
    # In Python bindings pass the explicit label.
    assert failed(lambda: checked_lt(T1, tis.label()))


def test_one_dimensional_tensor():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis])

    assert not failed(lambda: checked_lt(T1))


def test_one_dimensional_tensor_with_correct_string_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis])

    assert not failed(lambda: checked_lt(T1, "i"))


def test_one_dimensional_tensor_with_correct_label_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis])

    assert not failed(lambda: checked_lt(T1, tis.label()))


def test_one_dimensional_tensor_with_wrong_string_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis])

    assert failed(lambda: checked_lt(T1, "i", "i"))


def test_one_dimensional_tensor_with_wrong_label_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis])

    assert failed(lambda: checked_lt(T1, tis.label(0), tis.label(1)))


def test_two_dimensional_tensor():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis, tis])

    assert not failed(lambda: checked_lt(T1))


def test_two_dimensional_tensor_with_correct_string_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis, tis])

    assert not failed(lambda: checked_lt(T1, "i", "j"))


def test_two_dimensional_tensor_with_correct_label_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis, tis])

    assert not failed(lambda: checked_lt(T1, tis.label(0), tis.label(1)))


def test_two_dimensional_tensor_with_smaller_string_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis, tis])

    assert failed(lambda: checked_lt(T1, "i"))


def test_two_dimensional_tensor_with_smaller_label_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis, tis])

    assert failed(lambda: checked_lt(T1, tis.label(0)))


def test_two_dimensional_tensor_with_larger_string_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis, tis])

    assert failed(lambda: checked_lt(T1, "i", "j", "k"))


def test_two_dimensional_tensor_with_larger_label_count():
    tis = make_tis(10)
    T1 = tamm.TensorDouble([tis, tis])

    assert failed(lambda: checked_lt(T1, tis.label(0), tis.label(1), tis.label(2)))


def test_two_dimensional_dependent_index_tensor():
    is_ = tamm.IndexSpace(tamm.range(1))
    tis = tamm.TiledIndexSpace(is_)

    is2 = tamm.IndexSpace(tamm.range(5))

    dep_space_relation = {
        (0,): is2,
    }

    dis = tamm.IndexSpace([tis], dep_space_relation)
    tdis = tamm.TiledIndexSpace(dis)

    i, j = tis_labels(tis, 2)
    a, b = tis_labels(tdis, 2)

    # Tensor construction tests.
    assert failed(lambda: tamm.TensorDouble([tdis, tdis]))

    assert failed(lambda: tamm.TensorDouble([a(i), j]))

    assert not failed(lambda: tamm.TensorDouble([i(a), j]))

    # Labeled tensor tests.
    T1 = tamm.TensorDouble([a(i), i])

    assert not failed(lambda: checked_lt(T1))

    assert not failed(lambda: checked_lt(T1, b(j), j))

    assert failed(lambda: checked_lt(T1, "x"))

    assert failed(lambda: checked_lt(T1, a))

    assert failed(lambda: checked_lt(T1, i))

    assert not failed(lambda: checked_lt(T1, "x", "y"))

    assert not failed(lambda: checked_lt(T1, a, i))

    assert failed(lambda: checked_lt(T1, i, a(i)))

    assert failed(lambda: checked_lt(T1, a(i)))

    assert failed(lambda: checked_lt(T1, i))

    assert failed(lambda: checked_lt(T1, i(a)))

    assert failed(lambda: checked_lt(T1, a(a)))

    assert failed(lambda: checked_lt(T1, i(i)))
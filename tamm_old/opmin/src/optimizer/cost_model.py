from error import OptimizerError
from ast.absyn import Array, Addition, Multiplication


def countOpCost(e, index_tab, volatile_tab, iteration):
    return __countOpCostExp(e, index_tab, volatile_tab, iteration)


def __countOpCostExp(e, index_tab, volatile_tab, iteration):
    if (isinstance(e, Array)):
        return __countOpCostArray(e, index_tab, volatile_tab, iteration)
    elif (isinstance(e, Addition)):
        return __countOpCostAddition(e, index_tab, volatile_tab, iteration)
    elif (isinstance(e, Multiplication)):
        return __countOpCostMultiplication(e, index_tab, volatile_tab, iteration)
    else:
        raise OptimizerError('%s: unknown expression' % __name__)


def __countOpCostArray(e, index_tab, volatile_tab, iteration):
    is_volatile = volatile_tab.has_key(e.name)
    if (e.coef == 1.0 or e.coef == -1.0 or e.coef == 1 or e.coef == -1):
        return (0.0, is_volatile)
    else:
        arr_size = __getNumOfElements(index_tab, e.sym_groups)
        total = arr_size
        if (is_volatile):
            total *= iteration
        return (total, is_volatile)


def __countOpCostAddition(e, index_tab, volatile_tab, iteration):
    if (e.coef != 1):
        raise OptimizerError('%s: coefficient of an addition must be 1' % __name__)

    is_volatile = False
    subexps_cost = 0
    for se in e.subexps:
        (cur_cost, cur_is_volatile) = __countOpCostExp(se, index_tab, volatile_tab, iteration)
        is_volatile = is_volatile or cur_is_volatile
        subexps_cost += cur_cost

    loops = __getNumOfElements(index_tab, e.sym_groups)
    add_ops = len(e.subexps) - 1
    total = loops * add_ops

    if (is_volatile):
        total *= iteration
    total += subexps_cost

    if len(e.vsym_groups) > 1:
        vlen = len(e.vsym_groups[0])
        if vlen == 2:
            total /= 2
        elif vlen > 2:
            total /= __factorial(vlen)

    return (total, is_volatile)


def __countOpCostMultiplication(e, index_tab, volatile_tab, iteration):

    is_volatile = False

    subexps_cost = 0
    for se in e.subexps:
        (cur_cost, cur_is_volatile) = __countOpCostExp(se, index_tab, volatile_tab, iteration)
        is_volatile = is_volatile or cur_is_volatile
        subexps_cost += cur_cost

    outer_loops = __getNumOfElements(index_tab, e.sym_groups)
    inner_loops = __getNumOfElements(index_tab, e.sum_sym_groups)
    perms = __getNumOfPermuts(index_tab, e.ops)

    mult_ops = (len(e.subexps) - 1) * perms
    if (inner_loops == 1):
        add_ops = perms - 1
    else:
        add_ops = perms

    total = outer_loops * inner_loops * (mult_ops + add_ops)

    sum_coef = 1
    for g in e.sum_sym_groups:
        sum_coef *= __factorial(len(g))
    coef = (sum_coef * e.coef) / perms

    if (coef != 1 and coef != -1 and coef != 1.0 and coef != -1.0):
        total += outer_loops

    if (is_volatile):
        total *= iteration
    total += subexps_cost

    if len(e.vsym_groups) > 1:
        vlen = len(e.vsym_groups[0])
        if vlen == 2:
            total /= 4
        elif vlen > 2:
            total /= __factorial(vlen)

    return (total, is_volatile)


def __getNumOfElements(index_tab, sym_groups):
    t = 1.0
    for g in sym_groups:
        r = index_tab[g[0]]
        t *= __binomial(r, len(g))
    return t


def __getNumOfPermuts(index_tab, ops):
    t = 1.0
    for o in ops:
        total_lg = 0
        lo_fact = 1.0
        for g in o.sym_groups:
            lg = len(g)
            total_lg += lg
            lo_fact *= __factorial(lg)
        up_fact = __factorial(total_lg)
        t *= (up_fact / lo_fact)
    return t


__storage_table = {}


def __factorial(n):
    if n == 1:
        return 1.0

    if (__storage_table.has_key(n)):
        return __storage_table[n]

    t = 1.0
    while n > 1:
        t *= n
        n -= 1
    __storage_table[n] = t
    return t


def __binomial(n, k):
    t = 1.0
    for i in range(k):
        t *= (n - i)
    t /= __factorial(k)
    return t

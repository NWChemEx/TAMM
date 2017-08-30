from ast.absyn import (
    ArrayDecl,
    AssignStmt,
    Array,
    Addition,
    Multiplication,
)
from ast.absyn_lib import renameIndices, buildRenamingTable, extendRenamingTable
from math import gcd


def updateFAC(comp_elem, stmt_tab, index_tab, volatile_tab, iteration, range_tab):
    inter_names = stmt_tab.getAllIntermediateNames()

    editList = []
    for n1 in inter_names:
        stmt = stmt_tab.arr_map[n1]
        if (isinstance(stmt.rhs, Addition)):
            add = stmt_tab.arr_map[n1]
            l_name = add.rhs.subexps[0].name
            r_name = add.rhs.subexps[1].name

            if l_name == r_name:
                continue

            l_inds = add.rhs.subexps[0].upper_inds + add.rhs.subexps[0].lower_inds
            r_inds = add.rhs.subexps[1].upper_inds + add.rhs.subexps[1].lower_inds
            l_coef = add.rhs.subexps[0].coef
            r_coef = add.rhs.subexps[1].coef

            if l_name and r_name in stmt_tab.arr_map.keys():

                left_o = stmt_tab.arr_map[l_name]
                right_o = stmt_tab.arr_map[r_name]

                if isinstance(left_o.rhs, Multiplication) and isinstance(right_o.rhs, Multiplication):

                    left = AssignStmt(left_o.lhs, left_o.rhs.replicate())
                    from_inds = left.lhs.upper_inds + left.lhs.lower_inds
                    ren_tab = buildRenamingTable(from_inds, l_inds)
                    renameIndices(left.rhs, ren_tab)

                    right = AssignStmt(right_o.lhs, right_o.rhs.replicate())
                    from_inds2 = right.lhs.upper_inds + right.lhs.lower_inds
                    ren_tab2 = buildRenamingTable(from_inds2, r_inds)
                    renameIndices(right.rhs, ren_tab2)

                    [fact_mult, add, new_i, new_mult] = factTwoMultOneLevel(
                        left, right, l_coef, r_coef, stmt_tab, index_tab, volatile_tab, iteration)

                    if fact_mult is not None and add is not None:
                        stmt_tab.arr_map[n1].rhs = fact_mult
                        editList.append(n1)
                        editList.append(new_i.lhs.name)
                    else:  # search for second level
                        continue
                        [inter1_info, inter2_info, inter3] = factTwoMultTwoLevel(
                            left, right, l_coef, r_coef, stmt_tab, index_tab, volatile_tab, iteration)

                        if inter1_info is not None:
                            stmt_tab.arr_map[n1].rhs = inter3
                            editList.append(inter2_info.lhs.name)
                            editList.append(inter1_info.lhs.name)
                            editList.append(n1)

    com_inter_names = comp_elem.getAllInterNames()

    new_elems = []
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt)):
            if (isinstance(e.rhs, Array)):
                pass
            else:
                if e.lhs.name in editList:
                    e.rhs = stmt_tab.arr_map[e.lhs.name].rhs
                    new_elems.append(e)
                    continue
        new_elems.append(e)

    for ed in editList:
        if ed not in com_inter_names:
            lhs = stmt_tab.arr_map[ed].lhs
            rhs = stmt_tab.arr_map[ed].rhs
            new_elems.append(AssignStmt(lhs, rhs))
            new_elems.append(__createArrayDecl(lhs, range_tab))

    comp_elem.elems = new_elems


def factTwoMultTwoLevel(mult1, mult2, l_coef, r_coef, stmt_tab, index_tab, volatile_tab, iteration):

    inter1 = None
    inter2 = None
    inter3 = None
    inter1_info = None
    add_info = None
    ext_inds = mult1.lhs.upper_inds + mult1.lhs.lower_inds

    firstlevel = mult1.rhs.subexps + mult2.rhs.subexps
    (m1, sym1) = __symbolizeMult(mult1.rhs)
    (m2, sym2) = __symbolizeMult(mult2.rhs)
    firstlevel_sym = sym1+sym2

    comb = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ]

    for x in comb:
        find_factor = False
        a1 = firstlevel[x[0]]  # for expand
        a2 = firstlevel[x[1]]
        b1 = firstlevel[x[2]]  # for compare
        b1_sym = firstlevel_sym[x[2]]
        b2 = firstlevel[x[3]]

        if not a1.name.startswith('_a'):
            continue  # skip expanding primitive tensor
        if isinstance(stmt_tab.arr_map[a1.name].rhs, Addition):
            continue  # skip expanding addition tensor

        mult3 = stmt_tab.arr_map[a1.name]  # a1 body

        # renaming indices for mult3
        from_inds = mult3.lhs.upper_inds + mult3.lhs.lower_inds
        to_inds = a1.upper_inds + a1.lower_inds
        ren_tab = buildRenamingTable(from_inds, to_inds)
        renameIndices(mult3.rhs, ren_tab)

        a3 = mult3.rhs.subexps[0]  # left child of a1
        a4 = mult3.rhs.subexps[1]  # right child of a1

        inds = mult3.rhs.upper_inds + mult3.rhs.lower_inds
        for i in inds:
            if i not in ext_inds and i not in mult3.rhs.sum_inds:
                mult3.rhs.sum_inds += i

        (m3, sym3) = __symbolizeMult(mult3.rhs)
        (a3_sym, a4_sym) = sym3

        if a3_sym == b1_sym:
            find_factor = True
            a5 = a4
        elif a4_sym == b1_sym:
            find_factor = True
            a5 = a3

        if find_factor:
            if x[0] > 1:
                temp = l_coef
                l_coef = r_coef
                r_coef = temp

            inter1 = Multiplication([a5, a2])
            inter1_info = stmt_tab.getInfoForExp(inter1, index_tab, volatile_tab, iteration)
            mult = inter1_info.lhs.replicate()

            inter2 = Addition([mult, b2])
            mult.coef = l_coef
            b2.coef = r_coef
            add_info = stmt_tab.getInfoForExp(inter2, index_tab, volatile_tab, iteration)
            add_i_arr = add_info.lhs.replicate()
            ren_tab = buildRenamingTable(add_info.from_inds, add_info.to_inds)
            renameIndices(add_i_arr, ren_tab)

            inter3 = Multiplication([add_i_arr, b1])
            # inter3_info = stmt_tab.getInfoForExp(inter3, index_tab, volatile_tab, iteration)

            break

    return [inter1_info, add_info, inter3]


def factTwoMultOneLevel(mult1, mult2, l_coef, r_coef, stmt_tab, index_tab, volatile_tab, iteration):

    (m1, sym1) = __symbolizeMult(mult1.rhs)
    (m2, sym2) = __symbolizeMult(mult2.rhs)
    len1 = len(sym1)
    len2 = len(sym2)
    fact_mult = None
    add = None
    add_info = None
    fact_mult_info = None
    g = 1

    for i1 in range(0, len1):
        for i2 in range(0, len2):
            if (sym1[i1] == sym2[i2]):
                right1 = [se.replicate() for se in mult1.rhs.subexps]
                left1 = right1.pop(i1)
                right2 = [se.replicate() for se in mult2.rhs.subexps]
                left2 = right2.pop(i2)

                if (len(right1) == 1):
                    right1 = right1[0]
                if (len(right2) == 1):
                    right2 = right2[0]

                from_inds = left2.upper_inds + left2.lower_inds
                to_inds = left1.upper_inds + left1.lower_inds
                ren_tab = buildRenamingTable(from_inds, to_inds)
                mult_inds = mult2.rhs.upper_inds + mult2.rhs.lower_inds + mult2.rhs.sum_inds
                ren_tab = extendRenamingTable(ren_tab, set(mult_inds) - set(from_inds))
                renameIndices(left2, ren_tab)
                renameIndices(right2, ren_tab)

                sym_groups = mult1.rhs.sym_groups
                factor = left1

                [g, c1, c2] = gcd(l_coef, r_coef)

                right1.multiplyCoef(c1)
                right2.multiplyCoef(c2)

                multi_term = [right1, right2]
                if right1.name == right2.name:
                    if right1.coef < right2.coef:
                        multi_term = [right2, right1]
                if right1.name > right2.name:
                    multi_term = [right2, right1]

                add = Addition(multi_term)
                add_info = stmt_tab.getInfoForExp(add, index_tab, volatile_tab, iteration)
                add_i_arr = add_info.lhs.replicate()
                ren_tab = buildRenamingTable(add_info.from_inds, add_info.to_inds)
                renameIndices(add_i_arr, ren_tab)

                fact_mult = Multiplication([factor, add_i_arr])
                fact_mult.coef = g
                fact_mult.setOps(sym_groups)
                fact_mult_info = stmt_tab.getInfoForExp(fact_mult, index_tab, volatile_tab, iteration)

    return [fact_mult, add, add_info, fact_mult_info]


def collectFactor(term, stmt_tab, sum_ref):
    factors = []
    if isinstance(term, Array):
        return factors
    elif isinstance(term.rhs, Addition):
        return factors
    elif isinstance(term.rhs, Multiplication):
        sub1 = term.rhs.subexps[0]
        sub2 = term.rhs.subexps[1]
        term.rhs.sum_inds.extend(sum_ref)
        (m1, sym1) = __symbolizeMult(term.rhs)
        factors.extend(sym1)
        if sub1.name in stmt_tab.arr_map.keys():
            factors.extend(collectFactor(stmt_tab.arr_map[sub1.name], stmt_tab, sum_ref))
        if sub2.name in stmt_tab.arr_map.keys():
            factors.extend(collectFactor(stmt_tab.arr_map[sub2.name], stmt_tab, sum_ref))

    return factors


def updateCSE(comp_elem, stmt_tab):

    inter_names = stmt_tab.getAllIntermediateNames()

    removeList = []
    while len(inter_names) != 0:
        n1 = inter_names.pop(0)

        if n1 in removeList:
            continue

        can_forms_info = stmt_tab.getCf(n1)

        [n1_cf, to_inds1] = can_forms_info[0]
        for n2 in inter_names:
            can_forms_info = stmt_tab.getCf(n2)

            if can_forms_info is None:
                continue

            [n2_cf, to_inds2] = can_forms_info[0]
            if n1_cf == n2_cf:
                special = stmt_tab.arr_map[n1].rhs
                if special.subexps[0].name == special.subexps[1].name:  # 967 special case: y = 2x + x
                    if special.subexps[0].name.startswith('_a'):
                        comp_elem.modifyAddition(n1)
                        comp_elem.checkAddition()
                        comp_elem.checkMultiplication()
                        array1 = stmt_tab.arr_map[n1].lhs
                        array2 = stmt_tab.arr_map[n2].lhs

                removeList.append(n2)
                array1 = stmt_tab.arr_map[n1].lhs
                array2 = stmt_tab.arr_map[n2].lhs
                __replaceCompElem(array1, array2, comp_elem, stmt_tab, to_inds1)
                if array1.name != array2.name:
                    stmt_tab.updateSame(array1, array2)

    comp_elem.removeElems(removeList)


def refineTop(comp_elem, stmt_tab, index_tab, volatile_tab, iteration, range_tab):
    editList = []

    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt)):
            if (isinstance(e.rhs, Array)):
                pass
            elif (len(e.rhs.subexps) > 2):  # result assign stmt: hbar
                inter = [se.replicate() for se in e.rhs.subexps]
                for x in range(0, len(inter)):
                    le = inter[x]
                    if not le.name.startswith('_a'):
                        continue  # primitive tensor
                    for y in range(x, len(inter)):
                        rt = inter[y]
                        if not rt.name.startswith('_a'):
                            continue  # primitive tensor
                        if le.name == rt.name:
                            continue  # same tensor

                        l_name = le.name
                        r_name = rt.name
                        l_inds = le.upper_inds + le.lower_inds
                        r_inds = rt.upper_inds + rt.lower_inds
                        l_coef = le.coef
                        r_coef = rt.coef

                        if l_name and r_name in stmt_tab.arr_map.keys():

                            left_o = stmt_tab.arr_map[l_name]
                            right_o = stmt_tab.arr_map[r_name]

                            if isinstance(left_o.rhs, Multiplication) and isinstance(right_o.rhs, Multiplication):

                                left = AssignStmt(left_o.lhs, left_o.rhs.replicate())
                                from_inds = left.lhs.upper_inds + left.lhs.lower_inds
                                ren_tab = buildRenamingTable(from_inds, l_inds)
                                renameIndices(left.rhs, ren_tab)

                                right = AssignStmt(right_o.lhs, right_o.rhs.replicate())
                                from_inds2 = right.lhs.upper_inds + right.lhs.lower_inds
                                ren_tab2 = buildRenamingTable(from_inds2, r_inds)
                                renameIndices(right.rhs, ren_tab2)

                                [fact_mult, add, new_i, new_mult] = factTwoMultOneLevel(
                                    left, right, l_coef, r_coef, stmt_tab, index_tab, volatile_tab, iteration)

                                if (fact_mult is not None) and (add is not None):  # find factor

                                    inter2 = []
                                    for z in range(0, len(inter)):
                                        if z != x and z != y:
                                            inter2.append(inter[z].replicate())

                                    inter2.append(new_mult.lhs)

                                    e.rhs.subexps = inter2
                                    editList.append(new_i.lhs.name)
                                    editList.append(new_mult.lhs.name)

                                    com_inter_names = comp_elem.getAllInterNames()
                                    for ed in editList:
                                        if ed not in com_inter_names:
                                            lhs = stmt_tab.arr_map[ed].lhs
                                            rhs = stmt_tab.arr_map[ed].rhs
                                            comp_elem.elems.append(AssignStmt(lhs, rhs))
                                            comp_elem.elems.append(__createArrayDecl(lhs, range_tab))
                                    comp_elem.reArrange()

                                    return True

    return False


def __replaceCompElem(a1, a2, comp_elem, stmt_tab, to_inds):
    n1 = a1.name  # keep n1
    n2 = a2.name
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt)):
            if (isinstance(e.rhs, Array)):
                pass
            else:
                for x in range(0, len(e.rhs.subexps)):
                    if n2 == e.rhs.subexps[x].name:
                        e.rhs.subexps[x].name = n1
        else:
            pass


def __symbolizeMult(mult):
    coef = mult.coef
    arrs = []
    symbols = []

    for se in mult.subexps:
        (a, sy, s) = __getArraySymbol(se, mult.sum_inds)
        coef *= s
        arrs.append(a)
        symbols.append(sy)

    m = Multiplication(arrs, coef)
    m.setOps(mult.sym_groups)

    return (m, symbols)


def __createArrayDecl(arr, range_tab):
    upper_ranges = map(lambda x: range_tab[x], arr.upper_inds)
    lower_ranges = map(lambda x: range_tab[x], arr.lower_inds)
    inds = arr.upper_inds + arr.lower_inds
    sym_groups = [map(lambda x: inds.index(x), g) for g in arr.sym_groups]
    vsym_groups = [map(lambda x: inds.index(x), g) for g in arr.vsym_groups]
    for g in sym_groups:
        g.sort()
    sym_groups.sort()
    return ArrayDecl(arr.name, upper_ranges, lower_ranges, sym_groups, vsym_groups)


def __getArraySymbol(arr, sum_inds):
    r_arr = arr.replicate()
    sign = 1

    inds = r_arr.upper_inds + r_arr.lower_inds
    from_inds = []
    to_inds = []
    for g in r_arr.sym_groups:
        original_inds = filter(lambda x: x in g, inds)
        exts = []
        sums = []
        for i in original_inds:
            if (i in sum_inds):
                sums.append(i)
            else:
                exts.append(i)
        exts.sort()
        ordered_inds = exts + sums

        from_inds.extend(original_inds)
        to_inds.extend(ordered_inds)
        sign *= __countParitySign(original_inds, ordered_inds)

    ren_tab = buildRenamingTable(from_inds, to_inds)

    renameIndices(r_arr, ren_tab)

    inds = r_arr.upper_inds + r_arr.lower_inds

    symbol = ''
    symbol += r_arr.name + '['
    for (i, ind) in enumerate(inds):
        if (i > 0):
            symbol += ','
        if (ind in sum_inds):
            symbol += '#i'
        else:
            symbol += str(ind)
    symbol += ']'

    return (r_arr, symbol, sign)


def __countParitySign(ls1, ls2):
    ls2 = ls2[:]
    parity = 0
    for i in range(len(ls1)):
        if (ls1[i] != ls2[i]):
            swap_i = ls2.index(ls1[i])
            ls2[swap_i] = ls2[i]
            ls2[i] = ls1[i]
            parity += 1
    if (parity % 2 == 0):
        return 1
    else:
        return -1

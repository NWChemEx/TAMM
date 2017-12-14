from error import OptimizerError
from ast.absyn import Array, Addition, Multiplication
from ast.absyn_lib import buildRenamingTable


def canonicalizeExp(e):
    if (isinstance(e, Array)):
        return __canonicalizeArray(e)
    elif (isinstance(e, Addition)):
        return __canonicalizeAddition(e)
    elif (isinstance(e, Multiplication)):
        return __canonicalizeMultiplication(e)
    else:
        raise OptimizerError('%s: unknown expression' % __name__)


def __canonicalizeArray(e):

    # remember important elements
    arr_sym_groups = [g[:] for g in e.sym_groups]
    arr_coef = e.coef
    arr_name = e.name
    arr_upper_inds = e.upper_inds[:]
    arr_lower_inds = e.lower_inds[:]

    # get all possible index swaps
    coef_inds_info = [(arr_coef, arr_upper_inds, arr_lower_inds)]
    swap_groups = filter(lambda x: len(x) > 1, arr_sym_groups)
    for g in swap_groups:
        # permuts = __getAllPermutations(g)
        permuts = __getSortPermutationRepresent(g)
        signs = map(lambda x: __countParitySign(g, x), permuts)
        ren_tabs = map(lambda x: buildRenamingTable(g, x), permuts)
        new_coef_inds_info = []
        for (a_c, a_uis, a_lis) in coef_inds_info:
            for (s, r, p) in zip(signs, ren_tabs, permuts):
                n_a_c = a_c * s
                n_a_uis = map(lambda x: r.get(x, x), a_uis)
                n_a_lis = map(lambda x: r.get(x, x), a_lis)
                new_coef_inds_info.append((n_a_c, n_a_uis, n_a_lis))
        coef_inds_info = new_coef_inds_info

    # compute canonical forms info
    prefix = '#e'
    can_forms_tab = {}
    can_forms_info = []

    iter_info = []
    for (a_c, a_uis, a_lis) in coef_inds_info:
        iter_info.append((a_c, arr_name, a_uis, a_lis))
    for (a_c, a_name, a_uis, a_lis) in iter_info:
        ext_uis = a_uis
        ext_lis = a_lis

        from_inds = []
        to_inds = []
        num = 1
        for i in (a_uis + a_lis):
            from_inds.append(i)
            to_inds.append(prefix + str(num))
            num += 1
        ren_tab = buildRenamingTable(from_inds, to_inds)

        a_uis = map(lambda x: ren_tab.get(x, x), a_uis)
        a_lis = map(lambda x: ren_tab.get(x, x), a_lis)

        s = ''
        s += str(a_c) + ' * ' + a_name + str(a_uis) + str(a_lis)

        if (not can_forms_tab.has_key(s)):
            can_forms_tab[s] = None
            can_forms_info.append((s, ext_uis + ext_lis))

    # return canonical forms info
    return can_forms_info


def __canonicalizeAddition(e):

    # check assumption
    if (len(e.subexps) != 2):
        raise OptimizerError('%s: number of addition operands must be exactly two' % __name__)
    for se in e.subexps:
        if (not isinstance(se, Array)):
            raise OptimizerError('%s: addition operands must all be arrays' % __name__)

    # remember important elements
    add_sym_groups = [g[:] for g in e.sym_groups]
    ses_info = []
    for se in e.subexps:
        ses_info.append((se.coef*1.0, se.name, se.upper_inds[:], se.lower_inds[:]))
    subexps_infos = [ses_info]

    # get all possible index swaps
    swap_groups = []
    for i in range(0, len(e.subexps)):
        swap_groups.extend(map(lambda x: (i, x), filter(lambda x: len(x) > 1, add_sym_groups)))
    for (t, g) in swap_groups:
        # permuts = __getAllPermutations(g)
        permuts = __getSortPermutationRepresent(g)
        signs = map(lambda x: __countParitySign(g, x), permuts)
        ren_tabs = map(lambda x: buildRenamingTable(g, x), permuts)
        new_subexps_infos = []
        for ses_info in subexps_infos:
            for (s, r) in zip(signs, ren_tabs):
                n_ses_info = []
                for (i, (a_c, a_n, a_uis, a_lis)) in enumerate(ses_info):
                    if (t == i):
                        n_a_c = a_c * s
                        n_a_n = a_n
                        n_a_uis = map(lambda x: r.get(x, x), a_uis)
                        n_a_lis = map(lambda x: r.get(x, x), a_lis)
                    else:
                        n_a_c = a_c
                        n_a_n = a_n
                        n_a_uis = a_uis[:]
                        n_a_lis = a_lis[:]
                    n_ses_info.append((n_a_c, n_a_n, n_a_uis, n_a_lis))
                new_subexps_infos.append(n_ses_info)
        subexps_infos = new_subexps_infos

    # compute canonical forms info
    prefix = '#e'
    can_forms_tab = {}
    can_forms_info = []

    iter_info = []
    for ses_info in subexps_infos:
        # for p_ses_info in __getAllPermutations(ses_info):
        for p_ses_info in __getSortPermutationRepresent(ses_info):
            p_ses_info = [(a_c, a_n, a_uis[:], a_lis[:]) for (a_c, a_n, a_uis, a_lis) in p_ses_info]
            iter_info.append(p_ses_info)

    for ses_info in iter_info:
        (a0_c, a0_n, a0_uis, a0_lis) = ses_info[0]
        ext_uis = a0_uis
        ext_lis = a0_lis

        from_inds = []
        to_inds = []
        num = 1
        for i in (a0_uis + a0_lis):
            from_inds.append(i)
            to_inds.append(prefix + str(num))
            num += 1
        ren_tab = buildRenamingTable(from_inds, to_inds)

        s = ''
        for (i, (a_c, a_n, a_uis, a_lis)) in enumerate(ses_info):
            if (i > 0):
                s += ' + '
            a_uis = map(lambda x: ren_tab.get(x, x), a_uis)
            a_lis = map(lambda x: ren_tab.get(x, x), a_lis)
            s += str(a_c) + ' * ' + a_n + str(a_uis) + str(a_lis)

        if (not can_forms_tab.has_key(s)):
            can_forms_tab[s] = None
            can_forms_info.append((s, ext_uis + ext_lis))

    # return canonical forms info
    return can_forms_info


def __canonicalizeMultiplication(e):

    # check assumption
    if (len(e.subexps) > 2):
        raise OptimizerError('%s: number of multiplication operands must be exactly two' % __name__)
    if (
        (not isinstance(e.subexps[0], Array)) or (not isinstance(e.subexps[1], Array))
    ):
        raise OptimizerError('%s: multiplication operands must all be arrays' % __name__)

    # remember important elements
    mult_coef = e.coef * 1.0
    mult_sum_inds = e.sum_inds[:]
    mult_sym_groups = [g[:] for g in e.sym_groups]
    arr1_name = e.subexps[0].name   # tensor 1 name
    arr1_upper_inds = e.subexps[0].upper_inds[:]
    arr1_lower_inds = e.subexps[0].lower_inds[:]
    arr1_sym_groups = [g[:] for g in e.subexps[0].sym_groups]
    arr2_name = e.subexps[1].name   # tensor 2 name
    arr2_upper_inds = e.subexps[1].upper_inds[:]
    arr2_lower_inds = e.subexps[1].lower_inds[:]
    arr2_sym_groups = [g[:] for g in e.subexps[1].sym_groups]

    # get all possible index swaps
    ARR1 = 1
    ARR2 = 2
    MULT = 3
    coef_inds_info = [(mult_coef, arr1_upper_inds, arr1_lower_inds, arr2_upper_inds, arr2_lower_inds)]

    swap_groups = map(lambda x: (ARR1, x), filter(lambda x: len(x) > 1, arr1_sym_groups))
    swap_groups += map(lambda x: (ARR2, x), filter(lambda x: len(x) > 1, arr2_sym_groups))
    swap_groups += map(lambda x: (MULT, x), filter(lambda x: len(x) > 1, mult_sym_groups))

    for (t, g) in swap_groups:  # t = type   g = group
        # permuts = __getAllPermutations(g) #<-- original by Albert
        permuts = __getSortPermutationRepresent(g)

        signs = map(lambda x: __countParitySign(g, x), permuts)
        ren_tabs = map(lambda x: buildRenamingTable(g, x), permuts)
        new_coef_inds_info = []
        for (m_c, a1_uis, a1_lis, a2_uis, a2_lis) in coef_inds_info:
            for (s, r, p) in zip(signs, ren_tabs, permuts):
                if (t == ARR1):
                    n_m_c = m_c * s
                    n_a1_uis = map(lambda x: r.get(x, x), a1_uis)
                    n_a1_lis = map(lambda x: r.get(x, x), a1_lis)
                    n_a2_uis = a2_uis[:]
                    n_a2_lis = a2_lis[:]
                elif (t == ARR2):
                    n_m_c = m_c * s
                    n_a1_uis = a1_uis[:]
                    n_a1_lis = a1_lis[:]
                    n_a2_uis = map(lambda x: r.get(x, x), a2_uis)
                    n_a2_lis = map(lambda x: r.get(x, x), a2_lis)
                else:
                    assert(t == MULT), '%s: unknown type' % __name__
                    n_m_c = m_c * s
                    n_a1_uis = map(lambda x: r.get(x, x), a1_uis)
                    n_a1_lis = map(lambda x: r.get(x, x), a1_lis)
                    n_a2_uis = map(lambda x: r.get(x, x), a2_uis)
                    n_a2_lis = map(lambda x: r.get(x, x), a2_lis)
                new_coef_inds_info.append((n_m_c, n_a1_uis, n_a1_lis, n_a2_uis, n_a2_lis))
        coef_inds_info = new_coef_inds_info

    ext_prefix = '#e'
    sum_prefix = '#i'
    can_forms_tab = {}
    can_forms_info = []
    iter_info = []

    for (m_c, a1_uis, a1_lis, a2_uis, a2_lis) in coef_inds_info:
        if arr1_name > arr2_name:
            iter_info.append((m_c, arr1_name, a1_uis, a1_lis, arr2_name, a2_uis, a2_lis))               # t1 * t2
        else:
            iter_info.append((m_c, arr2_name, a2_uis[:], a2_lis[:], arr1_name, a1_uis[:], a1_lis[:]))   # t2 * t1

    for (m_c, a1_name, a1_uis, a1_lis, a2_name, a2_uis, a2_lis) in iter_info:
        ext_uis = filter(lambda x: x not in mult_sum_inds, a1_uis + a2_uis)     # a1_uis + a2_uis - mult_sum_inds
        ext_lis = filter(lambda x: x not in mult_sum_inds, a1_lis + a2_lis)

        from_inds = []
        to_inds = []
        ext_num = 1
        sum_num = 1
        seen = []
        for i in (a1_uis + a1_lis + a2_uis + a2_lis):   # i = every ind
            if (i in mult_sum_inds):    # i = contraction ind
                if (i not in seen):
                    from_inds.append(i)
                    to_inds.append(sum_prefix + str(sum_num))   # h3 -> #i1
                    sum_num += 1
            else:
                from_inds.append(i)
                to_inds.append(ext_prefix + str(ext_num))   # h1 -> #e1
                ext_num += 1
            seen.append(i)

        ren_tab = buildRenamingTable(from_inds, to_inds)

        a1_uis = map(lambda x: ren_tab.get(x, x), a1_uis)
        a1_lis = map(lambda x: ren_tab.get(x, x), a1_lis)
        a2_uis = map(lambda x: ren_tab.get(x, x), a2_uis)
        a2_lis = map(lambda x: ren_tab.get(x, x), a2_lis)

        sym_groups = [map(lambda x: ren_tab.get(x, x), g) for g in mult_sym_groups]
        for g in sym_groups:
            g.sort()
        sym_groups.sort()

        s = ''
        s += str(sym_groups) + ' ' + str(m_c) + ' * '
        s += a1_name + str(a1_uis) + str(a1_lis) + ' * '
        s += a2_name + str(a2_uis) + str(a2_lis)

        if (not can_forms_tab.has_key(s)):
            can_forms_tab[s] = None
            can_forms_info.append((s, ext_uis + ext_lis))

    return can_forms_info


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


def __getAllPermutations(ls):
    if (len(ls) <= 1):
        return [ls[:]]
    else:
        permuts = []
        for e in ls:
            cur_ls = ls[:]
            cur_ls.remove(e)
            ps = __getAllPermutations(cur_ls)
            for p in ps:
                p.insert(0, e)
            permuts.extend(ps)
        return permuts


def __getSortPermutationRepresent(ls):
    if (len(ls) <= 1):
        return [ls[:]]
    else:
        permuts = []
        permuts.append(ls[:])
        permuts[0].sort()
        return permuts

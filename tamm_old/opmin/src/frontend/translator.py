from error import FrontEndError
import ast.absyn
import absyn


def translate(trans_unit):
    return ast.absyn.TranslationUnit([__translateCompElem(ce, {}) for ce in trans_unit.comp_elems])


def __translateCompElem(ce, array_tab):
    return ast.absyn.CompoundElem([__translateElem(e, array_tab) for e in ce.elems])


def __translateElem(e, array_tab):
    if (isinstance(e, absyn.Decl)):
        return __translateDecl(e, array_tab)
    elif (isinstance(e, absyn.Stmt)):
        return __translateStmt(e, array_tab)
    else:
        raise FrontEndError('%s: unknown element' % __name__)


def __translateDecl(d, array_tab):
    if (isinstance(d, absyn.RangeDecl)):
        return __translateRangeDecl(d, array_tab)
    elif (isinstance(d, absyn.IndexDecl)):
        return __translateIndexDecl(d, array_tab)
    elif (isinstance(d, absyn.ArrayDecl)):
        return __translateArrayDecl(d, array_tab)
    elif (isinstance(d, absyn.ExpandDecl)):
        return __translateExpandDecl(d, array_tab)
    elif (isinstance(d, absyn.VolatileDecl)):
        return __translateVolatileDecl(d, array_tab)
    elif (isinstance(d, absyn.IterationDecl)):
        return __translateIterationDecl(d, array_tab)
    else:
        raise FrontEndError('%s: unknown declaration' % __name__)


def __translateRangeDecl(d, array_tab):
    return ast.absyn.RangeDecl(d.name, d.value.value)


def __translateIndexDecl(d, array_tab):
    return ast.absyn.IndexDecl(d.name, d.range.name)


def __translateArrayDecl(d, array_tab):
    up_ranges = map(lambda x: x.name, d.upper_ranges)
    lo_ranges = map(lambda x: x.name, d.lower_ranges)
    partial_sgroups = [[int(i.value) for i in g] for g in d.sym_groups]
    full_sgroups = []
    for i in range(len(up_ranges + lo_ranges)):
        found = False
        for g in full_sgroups:
            if (i in g):
                found = True
        if found:
            continue
        found = False
        for g in partial_sgroups:
            if (i in g):
                found = True
                full_sgroups.append(g[:])
                break
        if found:
            continue
        full_sgroups.append([i])
    vsgroups = [[int(i.value) for i in g] for g in d.vsym_groups]
    array_tab[d.name] = (len(up_ranges), full_sgroups, vsgroups)
    return ast.absyn.ArrayDecl(d.name, up_ranges, lo_ranges, full_sgroups, vsgroups)


def __translateExpandDecl(d, array_tab):
    return ast.absyn.ExpandDecl(d.arr.name)


def __translateVolatileDecl(d, array_tab):
    return ast.absyn.VolatileDecl(d.arr.name)


def __translateIterationDecl(d, array_tab):
    return ast.absyn.IterationDecl(d.value.value)


def __translateStmt(s, array_tab):
    if (isinstance(s, absyn.AssignStmt)):
        return __translateAssignStmt(s, array_tab)
    else:
        raise FrontEndError('%s: unknown statement' % __name__)


def __translateAssignStmt(s, array_tab):
    return ast.absyn.AssignStmt(__translateExp(s.lhs, array_tab), __translateExp(s.rhs, array_tab))


def __translateExp(e, array_tab):
    if (isinstance(e, absyn.Parenth)):
        raise FrontEndError('%s: parenthesized expression must not exist during optimization' % __name__)
    elif (isinstance(e, absyn.NumConst)):
        return __translateNumConst(e, array_tab)
    elif (isinstance(e, absyn.Array)):
        return __translateArray(e, array_tab)
    elif (isinstance(e, absyn.Addition)):
        return __translateAddition(e, array_tab)
    elif (isinstance(e, absyn.Multiplication)):
        return __translateMultiplication(e, array_tab)
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def __translateNumConst(e, array_tab):
    if (e.coef != -1 and e.coef != 1):
        raise FrontEndError('%s: coefficient of a numerical constant must be -1 or 1' % __name__)
    return ast.absyn.Array(e.value, e.coef, [], [])


def __translateArray(e, array_tab):
    (num_of_up_ranges, num_sgroups, num_vsgroups) = array_tab[e.name]
    inds = map(lambda x: x.name, e.inds)
    up_inds = inds[:num_of_up_ranges]
    lo_inds = inds[num_of_up_ranges:]
    ind_sgroups = [map(lambda x: inds[x], g) for g in num_sgroups]
    ind_vsgroups = [map(lambda x: inds[x], g) for g in num_vsgroups]
    return ast.absyn.Array(e.name, e.coef, up_inds, lo_inds, ind_sgroups, ind_vsgroups)


def __translateAddition(e, array_tab):
    if (e.coef != 1):
        raise FrontEndError('%s: coefficient of an addition must be 1' % __name__)
    subexps = [__translateExp(se, array_tab) for se in e.subexps]
    return ast.absyn.Addition(subexps, e.coef)


def __translateMultiplication(e, array_tab):
    subexps = [__translateExp(se, array_tab) for se in e.subexps]
    return ast.absyn.Multiplication(subexps, e.coef)


def applySym(trans_unit):
    for ce in trans_unit.comp_elems:
        for e in ce.elems:
            if (isinstance(e, ast.absyn.AssignStmt)):
                __applySymAssignStmt(e)


def __applySymAssignStmt(e):
    e.rhs = __applySymExp(e.rhs, e.lhs.sym_groups)


def __applySymExp(e, global_sym_groups):
    if (isinstance(e, ast.absyn.Array)):
        return __applySymArray(e, global_sym_groups)
    elif (isinstance(e, ast.absyn.Addition)):
        return __applySymAddition(e, global_sym_groups)
    elif (isinstance(e, ast.absyn.Multiplication)):
        return __applySymMultiplication(e, global_sym_groups)
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def __applySymArray(e, global_sym_groups):
    return e


def __applySymAddition(e, global_sym_groups):
    subexps = [__applySymExp(se, global_sym_groups) for se in e.subexps]
    return ast.absyn.Addition(subexps)


def __applySymMultiplication(e, global_sym_groups):
    subexps = [__applySymExp(se, global_sym_groups) for se in e.subexps]
    m = ast.absyn.Multiplication(subexps, e.coef)
    m.setOps(global_sym_groups)
    return m

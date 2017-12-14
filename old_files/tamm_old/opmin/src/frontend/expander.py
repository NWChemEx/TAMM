from error import FrontEndError
from absyn import (
    Decl,
    RangeDecl,
    IndexDecl,
    ArrayDecl,
    ExpandDecl,
    VolatileDecl,
    IterationDecl,
    Stmt,
    AssignStmt,
    Parenth,
    NumConst,
    Array,
    Addition,
    Multiplication
)
from absyn_lib import getIndices, getSumIndices


def expand(trans_unit):
    trans_unit.comp_elems = [__expandCompElem(ce, __SymbolTable()) for ce in trans_unit.comp_elems]


def __expandCompElem(comp_elem, symtab):
    expanded_elems = []
    for e in comp_elem.elems:
        ea = __expandElem(e, symtab)
        if ea is not None:
            expanded_elems.append(ea)
    comp_elem.elems = expanded_elems
    return comp_elem


def __expandElem(e, symtab):
    if (isinstance(e, Decl)):
        return __expandDecl(e, symtab)
    elif (isinstance(e, Stmt)):
        return __expandStmt(e, symtab)
    else:
        raise FrontEndError('%s: unknown element' % __name__)


def __expandDecl(d, symtab):
    if (isinstance(d, RangeDecl)):
        return __expandRangeDecl(d, symtab)
    elif (isinstance(d, IndexDecl)):
        return __expandIndexDecl(d, symtab)
    elif (isinstance(d, ArrayDecl)):
        return __expandArrayDecl(d, symtab)
    elif (isinstance(d, ExpandDecl)):
        return __expandExpandDecl(d, symtab)
    elif (isinstance(d, VolatileDecl)):
        return __expandVolatileDecl(d, symtab)
    elif (isinstance(d, IterationDecl)):
        return __expandIterationDecl(d, symtab)
    else:
        raise FrontEndError('%s: unknown declaration' % __name__)


def __expandRangeDecl(d, symtab):
    return d


def __expandIndexDecl(d, symtab):
    return d


def __expandArrayDecl(d, symtab):
    return d


def __expandExpandDecl(d, symtab):
    assert(d.arr.name not in symtab.expand_list), '%s: repeated expanded-array declaration' % __name__
    symtab.expand_list.append(d.arr.name)
    return d


def __expandVolatileDecl(d, symtab):
    return d


def __expandIterationDecl(d, symtab):
    return d


def __expandStmt(s, symtab):
    if (isinstance(s, AssignStmt)):
        return __expandAssignStmt(s, symtab)
    else:
        raise FrontEndError('%s: unknown statement' % __name__)


def __expandAssignStmt(s, symtab):
    s.rhs = __expandExp(s.rhs, symtab)
    symtab.assign_tab[s.lhs.name] = s

    if (s.lhs.name in symtab.expand_list):
        return None
    else:
        return s


def __expandExp(e, symtab):
    if (isinstance(e, Parenth)):
        return __expandParenth(e, symtab)
    elif (isinstance(e, NumConst)):
        return __expandNumConst(e, symtab)
    elif (isinstance(e, Array)):
        return __expandArray(e, symtab)
    elif (isinstance(e, Addition)):
        return __expandAddition(e, symtab)
    elif (isinstance(e, Multiplication)):
        return __expandMultiplication(e, symtab)
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def __expandParenth(e, symtab):
    e.exp = __expandExp(e.exp, symtab)
    return e


def __expandNumConst(e, symtab):
    return e


def __expandArray(e, symtab):
    if (e.name in symtab.expand_list and symtab.assign_tab.has_key(e.name)):
        a_stmt = symtab.assign_tab[e.name]
        expanded = a_stmt.rhs.replicate()
        expanded.multiplyCoef(e.coef)

        from_inames = map(lambda x: x.name, getIndices(a_stmt.lhs))
        to_inames = map(lambda x: x.name, getIndices(e))
        ren_tab = __buildRenamingTable(from_inames, to_inames)
        __renameIndices(expanded, ren_tab)

        return Parenth(expanded)
    else:
        return e


def __expandAddition(e, symtab):
    e.subexps = [__expandExp(se, symtab) for se in e.subexps]
    return e


def __expandMultiplication(e, symtab):
    e.subexps = [__expandExp(se, symtab) for se in e.subexps]
    return e


def __renameIndices(e, ren_tab):
    if (isinstance(e, Parenth)):
        __renameIndices(e.exp, ren_tab)
    elif (isinstance(e, NumConst)):
        pass
    elif (isinstance(e, Array)):
        for i in e.inds:
            i.name = ren_tab.get(i.name, i.name)
    elif (isinstance(e, Addition)):
        ren_tab = __filterRenamingTable(ren_tab, map(lambda x: x.name, getIndices(e)))
        for se in e.subexps:
            __renameIndices(se, ren_tab)
    elif (isinstance(e, Multiplication)):
        ren_tab = __filterRenamingTable(ren_tab, map(lambda x: x.name, getIndices(e)))
        ren_tab = __extendRenamingTable(ren_tab, map(lambda x: x.name, getSumIndices(e)))
        for se in e.subexps:
            __renameIndices(se, ren_tab)
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def __buildRenamingTable(from_inames, to_inames):
    ren_tab = {}
    for (f, t) in zip(from_inames, to_inames):
        if (f != t):
            ren_tab[f] = t
    return ren_tab


def __filterRenamingTable(ren_tab, inames):
    filtered_ren_tab = {}
    for (f, t) in ren_tab.iteritems():
        if (f in inames):
            filtered_ren_tab[f] = t
    return filtered_ren_tab


def __extendRenamingTable(ren_tab, sum_inames):
    ren_tab = ren_tab.copy()
    if (len(sum_inames) > 0):
        reversed_ren_tab = {}
        for (f, t) in ren_tab.iteritems():
            reversed_ren_tab[t] = f
        for i in sum_inames:
            cur_i = i
            visited = [i]
            while (reversed_ren_tab.has_key(cur_i)):
                cur_i = reversed_ren_tab[cur_i]
                if (cur_i in visited):
                    raise FrontEndError('%s: an illegal cycle is present in renaming table' % __name__)
                visited.append(cur_i)
            if (i != cur_i):
                ren_tab[i] = cur_i
    return ren_tab


class __SymbolTable:

    def __init__(self):
        self.expand_list = []
        self.assign_tab = {}

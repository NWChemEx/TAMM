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
    Identifier,
    Parenth,
    NumConst,
    Array,
    Addition,
    Multiplication
)
from absyn_lib import getIndices, getSumIndices


def flatten(trans_unit):
    trans_unit.comp_elems = [
        __flattenCompElem(ce, __SymbolTable(__VarNameGenerator('_i'))) for ce in trans_unit.comp_elems
    ]


def __flattenCompElem(comp_elem, symtab):
    flattened_elems = []
    for e in comp_elem.elems:
        flattened_elems.extend(__flattenElem(e, symtab))
    comp_elem.elems = flattened_elems
    return comp_elem


def __flattenElem(e, symtab):
    if (isinstance(e, Decl)):
        return __flattenDecl(e, symtab)
    elif (isinstance(e, Stmt)):
        return __flattenStmt(e, symtab)
    else:
        raise FrontEndError('%s: unknown element' % __name__)


def __flattenDecl(d, symtab):
    if (isinstance(d, RangeDecl)):
        return __flattenRangeDecl(d, symtab)
    elif (isinstance(d, IndexDecl)):
        return __flattenIndexDecl(d, symtab)
    elif (isinstance(d, ArrayDecl)):
        return __flattenArrayDecl(d, symtab)
    elif (isinstance(d, ExpandDecl)):
        return __flattenExpandDecl(d, symtab)
    elif (isinstance(d, VolatileDecl)):
        return __flattenVolatileDecl(d, symtab)
    elif (isinstance(d, IterationDecl)):
        return __flattenIterationDecl(d, symtab)
    else:
        raise FrontEndError('%s: unknown declaration' % __name__)


def __flattenRangeDecl(d, symtab):
    symtab.range_tab[d.name] = []
    return [d]


def __flattenIndexDecl(d, symtab):
    symtab.index_tab[d.name] = d.range.name
    symtab.range_tab[d.range.name].append(d.name)
    return [d]


def __flattenArrayDecl(d, symtab):
    return [d]


def __flattenExpandDecl(d, symtab):
    return [d]


def __flattenVolatileDecl(d, symtab):
    return [d]


def __flattenIterationDecl(d, symtab):
    return [d]


def __flattenStmt(s, symtab):
    if (isinstance(s, AssignStmt)):
        return __flattenAssignStmt(s, symtab)
    else:
        raise FrontEndError('%s: unknown statement' % __name__)


def __flattenAssignStmt(s, symtab):
    number_tab = {}
    numbered_rhs = __numberExp(s.rhs, number_tab, __VarNameGenerator('#'))
    flattened_rhs = __flattenExp(numbered_rhs)
    new_index_decls = []
    unnumbered_rhs = __unnumberExp(flattened_rhs, symtab, number_tab, new_index_decls)
    s.rhs = unnumbered_rhs
    return new_index_decls + [s]


def __flattenExp(e):
    if (isinstance(e, Parenth)):
        return __flattenParenth(e)
    elif (isinstance(e, NumConst)):
        return __flattenNumConst(e)
    elif (isinstance(e, Array)):
        return __flattenArray(e)
    elif (isinstance(e, Addition)):
        return __flattenAddition(e)
    elif (isinstance(e, Multiplication)):
        return __flattenMultiplication(e)
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def __flattenParenth(e):
    return __flattenExp(e.exp)


def __flattenNumConst(e):
    return e


def __flattenArray(e):
    return e


def __flattenAddition(e):
    flattened = []
    for se in e.subexps:
        f = __flattenExp(se)
        if (isinstance(f, Addition)):
            flattened.extend(f.subexps)
        else:
            flattened.append(f)
    e.subexps = flattened
    return e


def __flattenMultiplication(e):
    coef = 1
    flattened = []
    for se in e.subexps:
        f = __flattenExp(se)
        if (isinstance(f, Multiplication)):
            coef *= f.coef
            flattened.extend(f.subexps)
        else:
            flattened.append(f)
    e.multiplyCoef(coef)
    e.subexps = flattened

    unfact = e.subexps[0]
    for se in e.subexps[1:]:
        unfact = __unfactorize(unfact, se)
    unfact.multiplyCoef(e.coef)

    return unfact


def __numberExp(e, number_tab, vname_generator):
    if (isinstance(e, Parenth)):
        return __numberParenth(e, number_tab, vname_generator)
    elif (isinstance(e, NumConst)):
        return __numberNumConst(e, number_tab, vname_generator)
    elif (isinstance(e, Array)):
        return __numberArray(e, number_tab, vname_generator)
    elif (isinstance(e, Addition)):
        return __numberAddition(e, number_tab, vname_generator)
    elif (isinstance(e, Multiplication)):
        return __numberMultiplication(e, number_tab, vname_generator)
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def __numberParenth(e, number_tab, vname_generator):
    e.exp = __numberExp(e.exp, number_tab, vname_generator)
    return e


def __numberNumConst(e, number_tab, vname_generator):
    return e


def __numberArray(e, number_tab, vname_generator):
    return e


def __numberAddition(e, number_tab, vname_generator):
    e.subexps = [__numberExp(se, number_tab, vname_generator) for se in e.subexps]
    return e


def __numberMultiplication(e, number_tab, vname_generator):
    e.subexps = [__numberExp(se, number_tab, vname_generator) for se in e.subexps]

    sum_inames = map(lambda x: x.name, getSumIndices(e))
    number_inames = [vname_generator.generate() for i in range(0, len(sum_inames))]
    __renameIndices(e, dict(zip(sum_inames, number_inames)))

    tab = dict(zip(number_inames, sum_inames))
    number_tab.update(tab)

    return e


def __unnumberExp(e, symtab, number_tab, new_index_decls):
    if (isinstance(e, Parenth)):
        raise FrontEndError('%s: parenthesized expression must not exist during unnumbering process' % __name__)
    elif (isinstance(e, NumConst)):
        return __unnumberNumConst(e, symtab, number_tab, new_index_decls)
    elif (isinstance(e, Array)):
        return __unnumberArray(e, symtab, number_tab, new_index_decls)
    elif (isinstance(e, Addition)):
        return __unnumberAddition(e, symtab, number_tab, new_index_decls)
    elif (isinstance(e, Multiplication)):
        return __unnumberMultiplication(e, symtab, number_tab, new_index_decls)
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def __unnumberNumConst(e, symtab, number_tab, new_index_decls):
    return e


def __unnumberArray(e, symtab, number_tab, new_index_decls):
    return e


def __unnumberAddition(e, symtab, number_tab, new_index_decls):
    e.subexps = [__unnumberExp(se, symtab, number_tab, new_index_decls) for se in e.subexps]
    return e


def __unnumberMultiplication(e, symtab, number_tab, new_index_decls):
    for se in e.subexps:
        if (not (isinstance(se, NumConst) or isinstance(se, Array))):
            raise FrontEndError('%s: expression is not fully flattened' % __name__)

    range_tab = {}
    for (k, v) in symtab.range_tab.iteritems():
        range_tab[k] = v[:]

    ext_inames = map(lambda x: x.name, getIndices(e))
    sum_inames = map(lambda x: x.name, getSumIndices(e))
    to_inames = []

    for i in ext_inames:
        r = symtab.index_tab[i]
        range_tab[r].remove(i)

    for i in sum_inames:
        r = symtab.index_tab[number_tab[i]]
        if (len(range_tab[r]) == 0):
            t = symtab.vname_generator.generate()
            new_index_decls.append(IndexDecl(t, Identifier(r)))
            symtab.index_tab[t] = r
            symtab.range_tab[r].append(t)
        else:
            t = range_tab[r].pop(0)
        to_inames.append(t)

    __renameIndices(e, dict(zip(sum_inames, to_inames)))

    return e


def __unfactorize(e1, e2):
    if ((isinstance(e1, NumConst) or isinstance(e1, Array)) and (isinstance(e2, NumConst) or isinstance(e2, Array))):
        return Multiplication([e1, e2])
    elif ((isinstance(e1, NumConst) or isinstance(e1, Array)) and isinstance(e2, Addition)):
        e2.subexps = [__unfactorize(e1.replicate(), se) for se in e2.subexps]
        return e2
    elif ((isinstance(e1, NumConst) or isinstance(e1, Array)) and isinstance(e2, Multiplication)):
        e2.subexps = [e1] + e2.subexps
        return e2
    elif (isinstance(e1, Addition) and (isinstance(e2, NumConst) or isinstance(e2, Array))):
        e1.subexps = [__unfactorize(se, e2.replicate()) for se in e1.subexps]
        return e1
    elif (isinstance(e1, Addition) and isinstance(e2, Addition)):
        subexps = []
        for se1 in e1.subexps:
            for se2 in e2.subexps:
                subexps.append(__unfactorize(se1.replicate(), se2.replicate()))
        e1.subexps = subexps
        return e1
    elif (isinstance(e1, Addition) and isinstance(e2, Multiplication)):
        e1.subexps = [__unfactorize(se, e2.replicate()) for se in e1.subexps]
        return e1
    elif (isinstance(e1, Multiplication) and (isinstance(e2, NumConst) or isinstance(e2, Array))):
        e1.subexps = e1.subexps + [e2]
        return e1
    elif (isinstance(e1, Multiplication) and isinstance(e2, Addition)):
        e2.subexps = [__unfactorize(e1.replicate(), se) for se in e2.subexps]
        return e2
    elif (isinstance(e1, Multiplication) and isinstance(e2, Multiplication)):
        e1.subexps = e1.subexps + e2.subexps
        e1.multiplyCoef(e2.coef)
        return e1
    else:
        raise FrontEndError('%s: illegal operands types' % __name__)


def __renameIndices(e, ren_tab):
    if (isinstance(e, Parenth)):
        __renameIndices(e.exp, ren_tab)
    elif (isinstance(e, NumConst)):
        pass
    elif (isinstance(e, Array)):
        for i in e.inds:
            i.name = ren_tab.get(i.name, i.name)
    elif (isinstance(e, Addition)):
        for se in e.subexps:
            __renameIndices(se, ren_tab)
    elif (isinstance(e, Multiplication)):
        for se in e.subexps:
            __renameIndices(se, ren_tab)
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


class __VarNameGenerator:

    def __init__(self, prefix):
        self.num = 1
        self.prefix = prefix

    def generate(self):
        iname = self.prefix + str(self.num)
        self.num += 1
        return iname


class __SymbolTable:

    def __init__(self, vname_generator):
        self.range_tab = {}
        self.index_tab = {}
        self.vname_generator = vname_generator

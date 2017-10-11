from error import BackEndError
from ast.absyn import Decl, RangeDecl, IndexDecl, ArrayDecl, ExpandDecl, VolatileDecl,\
     IterationDecl, Stmt, AssignStmt, Array, Addition, Multiplication


def unparse(trans_unit):
    trans_unit = trans_unit.replicate()
    return '\n'.join(map(lambda x: '\n' + unparseCompElem(x), trans_unit.comp_elems))


def unparseCompElem(comp_elem):
    elems = __combineDecls(comp_elem.elems)

    s = '{'
    s += ''.join(map(lambda x: '\n ' + __unparseElem(x), elems))
    if (len(elems) > 0):
        s += '\n'
    s += '}'
    return s


def __unparseElem(e):
    if (isinstance(e, Decl)):
        return __unparseDecl(e)
    elif (isinstance(e, Stmt)):
        return __unparseStmt(e)
    else:
        raise BackEndError('%s: unknown element' % __name__)


def __unparseDecl(d):
    if (isinstance(d, RangeDecl)):
        return __unparseRangeDecl(d)
    elif (isinstance(d, IndexDecl)):
        return __unparseIndexDecl(d)
    elif (isinstance(d, ArrayDecl)):
        return __unparseArrayDecl(d)
    elif (isinstance(d, ExpandDecl)):
        return __unparseExpandDecl(d)
    elif (isinstance(d, VolatileDecl)):
        return __unparseVolatileDecl(d)
    elif (isinstance(d, IterationDecl)):
        return __unparseIterationDecl(d)
    else:
        raise BackEndError('%s: unknown declaration' % __name__)


def __unparseRangeDecl(d):
    s = ''
    s += 'range '
    s += ', '.join(map(lambda x: str(x.name), d.decls))
    s += ' = '
    s += str(d.value)
    s += ';'
    return s


def __unparseIndexDecl(d):
    s = ''
    s += 'index '
    s += ', '.join(map(lambda x: str(x.name), d.decls))
    s += ' = ' + str(d.range) + ';'
    return s


def __unparseArrayDecl(d):
    def __arrStr(a):
        s = ''
        s += str(a.name) + '(['
        s += ','.join(map(str, a.upper_ranges))
        s += ']['
        s += ','.join(map(str, a.lower_ranges))
        s += ']'
        sym = ''
        for g in a.sym_groups:
            if (len(g) > 1):
                sym += '(' + ','.join(map(lambda x: str(int(x)), g)) + ')'
        if (sym != ''):
            s += ':' + sym
        vsym = ''
        for g in a.vsym_groups:
            if (len(g) > 1):
                vsym += '<' + ','.join(map(lambda x: str(int(x)), g)) + '>'
        if (vsym != ''):
            s += ':' + vsym
        s += ')'
        return s

    s = ''
    s += 'array '
    s += ', '.join(map(__arrStr, d.decls))
    s += ';'
    return s


def __unparseExpandDecl(d):
    s = ''
    s += 'expand '
    s += ', '.join(map(lambda x: str(x.arr), d.decls))
    s += ';'
    return s


def __unparseVolatileDecl(d):
    s = ''
    s += 'volatile '
    s += ', '.join(map(lambda x: str(x.arr), d.decls))
    s += ';'
    return s


def __unparseIterationDecl(d):
    s = ''
    s += 'iteration = ' + str(d.value)
    s += ';'
    return s


def __unparseStmt(s):
    if (isinstance(s, AssignStmt)):
        return __unparseAssignStmt(s)
    else:
        raise BackEndError('%s: unknown statement' % __name__)


def __unparseAssignStmt(stmt):
    s = ''
    s += __unparseExp(stmt.lhs)
    s += ' = '
    s += __unparseExp(stmt.rhs)
    s += ';'
    return s


def __unparseExp(e):
    if (isinstance(e, Array)):
        return __unparseArray(e)
    elif (isinstance(e, Addition)):
        return __unparseAddition(e)
    elif (isinstance(e, Multiplication)):
        return __unparseMultiplication(e)
    else:
        raise BackEndError('%s: unknown expression' % __name__)


def __unparseArray(e):
    s = ''
    if (e.coef == -1):
        s += '-'
    elif (e.coef != -1 and e.coef != 1):
        s += '(' + str(e.coef) + ' * '
    s += str(e.name)
    if (len(e.upper_inds + e.lower_inds) > 0):
        s += '['
        s += ','.join(map(str, e.upper_inds + e.lower_inds))
        s += ']'
    if (e.coef != -1 and e.coef != 1):
        s += ')'
    return s


def __unparseAddition(e):
    s = ''
    if (e.coef == -1):
        s += '-'
    elif (e.coef != -1 and e.coef != 1):
        s += '(' + str(e.coef) + ' * '
    s += '('
    s += ' + '.join(map(__unparseExp, e.subexps))
    s += ')'
    if (e.coef != -1 and e.coef != 1):
        s += ')'
    return s


def __unparseMultiplication(e):
    def __opStr(o):
        s = ''
        s += 'S('
        for i, g in enumerate(o.sym_groups):
            if (i > 0):
                s += '|'
            s += '(' + ','.join(g) + ')'
        s += ')'
        return s

    s = ''
    if (e.coef == -1):
        s += '-'
    s += '('
    if (len(e.ops) > 0):
        s += ' * '.join(map(__opStr, e.ops)) + ' * '
    if (e.coef != -1 and e.coef != 1):
        s += str(e.coef) + ' * '
    s += ' * '.join(map(__unparseExp, e.subexps))
    s += ')'
    return s


def __combineDecls(elems):
    combined_decls = []
    prev = None
    for cur in elems:
        if (isinstance(cur, Decl)):
            cur.decls = [cur]

        if prev is None:
            prev = cur
            continue

        if (
            (isinstance(prev, RangeDecl) and isinstance(cur, RangeDecl) and prev.value == cur.value) or
            (isinstance(prev, IndexDecl) and isinstance(cur, IndexDecl) and prev.range == cur.range) or
            (isinstance(prev, ArrayDecl) and isinstance(cur, ArrayDecl)) or
            (isinstance(prev, ExpandDecl) and isinstance(cur, ExpandDecl)) or
            (isinstance(prev, VolatileDecl) and isinstance(cur, VolatileDecl))
        ):
            if (len(prev.decls) >= 5):
                combined_decls.append(prev)
                prev = cur
            else:
                prev.decls.append(cur)
        else:
            combined_decls.append(prev)
            prev = cur

    if prev is not None:
        combined_decls.append(prev)

    return combined_decls

from error import InputError, FrontEndError
from absyn import Decl, RangeDecl, IndexDecl, ArrayDecl, ExpandDecl, VolatileDecl, IterationDecl, \
     Stmt, AssignStmt, Parenth, NumConst, Array, Addition, Multiplication
from absyn_lib import getIndices
import ast.absyn


def semantCheck(trans_unit):
    for ce in trans_unit.comp_elems:
        __checkCompElem(ce, __SymbolTable())


def __checkCompElem(comp_elem, symtab):
    for e in comp_elem.elems:   # e = each line of the .eq
        __checkElem(e, symtab)


def __checkElem(e, symtab):
    if (isinstance(e, Decl)):   # e is a declaration, i.e. array c_om[O][M]
        __checkDecl(e, symtab)
    elif (isinstance(e, Stmt)):  # e is a statement: hbar[p3,p4,h1,h2] = (-0.5 * t_vo[p3,h3] * v_ovoo[h3,p4,h1,h2])
        __checkStmt(e, symtab)
    else:
        raise FrontEndError('%s: unknown element' % __name__)


def __checkDecl(d, symtab):
    if (isinstance(d, RangeDecl)):      # range O = 10.0;
        __checkRangeDecl(d, symtab)
    elif (isinstance(d, IndexDecl)):    # index h1 = O;
        __checkIndexDecl(d, symtab)
    elif (isinstance(d, ArrayDecl)):    # array f_ov([O][V]);
        __checkArrayDecl(d, symtab)
    elif (isinstance(d, ExpandDecl)):
        __checkExpandDecl(d, symtab)
    elif (isinstance(d, VolatileDecl)):
        __checkVolatileDecl(d, symtab)
    elif (isinstance(d, IterationDecl)):
        __checkIterationDecl(d, symtab)
    else:
        raise FrontEndError('%s: unknown declaration' % __name__)


def __checkRangeDecl(d, symtab):
    __verifyVarDecl(d.name, __getLineNo(d), symtab)
    if ((d.value.value % 1 != 0) or (d.value.value <= 0)):
        raise InputError('"%s" is not a positive integer' % d.value, __getLineNo(d.value))
    symtab.range_tab[d.name] = d.value.value


def __checkIndexDecl(d, symtab):
    __verifyVarDecl(d.name, __getLineNo(d), symtab)
    __verifyRangeRef(d.range.name, __getLineNo(d.range), symtab)
    symtab.index_tab[d.name] = d.range.name


def __checkArrayDecl(d, symtab):
    __verifyVarDecl(d.name, __getLineNo(d), symtab)
    for r in (d.upper_ranges + d.lower_ranges):
        __verifyRangeRef(r.name, __getLineNo(r), symtab)
    __verifySymGroups(d, symtab)
    symtab.array_tab[d.name] = ([r.name for r in d.upper_ranges], [r.name for r in d.lower_ranges])


def __checkExpandDecl(d, symtab):
    if (symtab.expand_tab.has_key(d.arr.name)):
        raise InputError('"%s" already exists in expansion list' % d.arr.name, __getLineNo(d.arr))
    __verifyArrayRefName(d.arr.name, __getLineNo(d.arr), symtab)
    symtab.expand_tab[d.arr.name] = d.arr.name


def __checkVolatileDecl(d, symtab):
    if (symtab.volatile_tab.has_key(d.arr.name)):
        raise InputError('"%s" already exists in volatile list' % d.arr.name, __getLineNo(d.arr))
    __verifyArrayRefName(d.arr.name, __getLineNo(d.arr), symtab)
    symtab.volatile_tab[d.arr.name] = d.arr.name


def __checkIterationDecl(d, symtab):
    if symtab.iteration is not None:
        raise InputError('iteration count is already defined', __getLineNo(d))
    if ((d.value.value % 1 != 0) or (d.value.value <= 0)):
        raise InputError('"%s" is not a positive integer' % d.value, __getLineNo(d.value))
    symtab.iteration = d.value.value


def __checkStmt(s, symtab):
    if (isinstance(s, AssignStmt)):
        __checkAssignStmt(s, symtab)
    else:
        raise FrontEndError('%s: unknown statement' % __name__)


def __checkAssignStmt(s, symtab):
    __checkExp(s.lhs, symtab)
    __checkExp(s.rhs, symtab)
    if (not isinstance(s.lhs, Array)):
        raise InputError('LHS of assignment must be an array reference', __getLineNo(s.lhs))
    elif (s.lhs.isNegative()):
        raise InputError('LHS array cannot be negative', __getLineNo(s.lhs))
    #print str(set(map(lambda x: x.name, getIndices(s.lhs)))) + "=" + str(set(map(lambda x: x.name, getIndices(s.rhs))))
    
    if (set(map(lambda x: x.name, getIndices(s.lhs))) != set(map(lambda x: x.name, getIndices(s.rhs)))):
        raise InputError('LHS and RHS of assignment must have equal index sets', __getLineNo(s))

    lhs_aref = list(__collectArrayRefs(s.lhs))[0]
    rhs_arefs = list(__collectArrayRefs(s.rhs))
    if (lhs_aref in rhs_arefs):
        raise InputError('array "%s" cannot be assigned after being previously referenced (line %s)' %
                         (lhs_aref, __getLineNo(s)), __getLineNo(s))
    for (cur_lhs_aref, cur_rhs_arefs, cur_line_no) in symtab.assign_seq:
        if (lhs_aref in cur_rhs_arefs):
            raise InputError('array "%s" cannot be assigned after being previously referenced (line %s)' %
                             (lhs_aref, cur_line_no), __getLineNo(s))
    symtab.assign_seq.append((lhs_aref, rhs_arefs, __getLineNo(s)))


def __checkExp(e, symtab):
    if (isinstance(e, Parenth)):
        __checkParenth(e, symtab)
    elif (isinstance(e, NumConst)):
        __checkNumConst(e, symtab)
    elif (isinstance(e, Array)):
        __checkArray(e, symtab)
    elif (isinstance(e, Addition)):
        __checkAddition(e, symtab)
    elif (isinstance(e, Multiplication)):
        __checkMultiplication(e, symtab)
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def __checkParenth(e, symtab):
    __checkExp(e.exp, symtab)


def __checkNumConst(e, symtab):
    pass


def __checkArray(e, symtab):
    __verifyArrayRef(e.name, __getLineNo(e), e.inds, symtab)
    inames = map(lambda x: x.name, getIndices(e))
    rnames = map(lambda x: symtab.index_tab[x], inames)
    (up_range_names, lo_range_names) = symtab.array_tab[e.name]
    istruct = up_range_names + lo_range_names
    if (istruct != rnames):
        raise InputError(
            'array reference "%s%s" must have index structure of "%s%s"' % (e.name, e.inds, e.name, istruct),
            __getLineNo(e)
        )
    for i in inames:
        if (inames.count(i) > 1):
            raise InputError(
                'indistinct index "%s" must not exist in array reference "%s%s"' % (i, e.name, e.inds),
                __getLineNo(e)
            )


def __checkAddition(e, symtab):
    for se in e.subexps:
        __checkExp(se, symtab)
    add_iname_set = set(map(lambda x: x.name, getIndices(e)))
    for se in e.subexps:
        op_iname_set = set(map(lambda x: x.name, getIndices(se)))
        if (add_iname_set != op_iname_set):
            raise InputError('subexpressions of an addition must have equal index sets', __getLineNo(e))


def __checkMultiplication(e, symtab):
    for se in e.subexps:
        __checkExp(se, symtab)
    inames = reduce(lambda x, y: x + y, [map(lambda x: x.name, getIndices(se)) for se in e.subexps], [])
    for i in inames:
        if (inames.count(i) > 2):
            raise InputError('summation index "%s" must occur exactly twice in a multiplication' % i, __getLineNo(e))


def __verifyVarDecl(name, line_no, symtab):
    if (symtab.range_tab.has_key(name) or symtab.index_tab.has_key(name) or symtab.array_tab.has_key(name)):
        raise InputError('"%s" is already defined' % name, line_no)


def __verifyRangeRef(name, line_no, symtab):
    if (not symtab.range_tab.has_key(name)):
        if (symtab.index_tab.has_key(name) or symtab.array_tab.has_key(name)):
            raise InputError('"%s" is not a range' % name, line_no)
        else:
            raise InputError('range "%s" is undefined' % name, line_no)


def __verifyIndexRef(name, line_no, symtab):
    if (not symtab.index_tab.has_key(name)):
        if (symtab.range_tab.has_key(name) or symtab.array_tab.has_key(name)):
            raise InputError('"%s" is not an index' % name, line_no)
        else:
            raise InputError('index "%s" is undefined' % name, line_no)


def __verifyArrayRef(name, line_no, inds, symtab):
    __verifyArrayRefName(name, line_no, symtab)
    for i in inds:
        __verifyIndexRef(i.name, __getLineNo(i), symtab)


def __verifyArrayRefName(name, line_no, symtab):
    if (not symtab.array_tab.has_key(name)):
        if (symtab.range_tab.has_key(name) or symtab.index_tab.has_key(name)):
            raise InputError('"%s" is not an array' % name, line_no)
        else:
            raise InputError('array "%s" is undefined' % name, line_no)


def __verifySymGroups(arr_decl, symtab):
    ranges = arr_decl.upper_ranges + arr_decl.lower_ranges
    for g in arr_decl.sym_groups:
        for ipos in g:
            __checkNumConst(ipos, symtab)
            if ((ipos.value % 1 != 0) or (ipos.value < 0)):
                raise InputError('"%s" is not a valid index position' % ipos.value, __getLineNo(ipos))
            if (ipos.value >= len(ranges)):
                raise InputError('index "%s" is out of range' % int(ipos.value), __getLineNo(ipos))
            all_indexes = map(lambda x: x.value, reduce(lambda x, y: x+y, arr_decl.sym_groups, []))
            if (all_indexes.count(ipos.value) > 1):
                raise InputError('index "%s" cannot occur more than once' % int(ipos.value), __getLineNo(ipos))
        sgroups_ranges = map(lambda x: ranges[int(x.value)].name, g)
        for r in sgroups_ranges:
            pass


def __collectArrayRefs(e):
    if (isinstance(e, Parenth)):
        return __collectArrayRefs(e.exp)
    elif (isinstance(e, NumConst)):
        return set([])
    elif (isinstance(e, Array)):
        return set([e.name])
    elif (isinstance(e, Addition)):
        return reduce(lambda x, y: x | y, [__collectArrayRefs(se) for se in e.subexps], set([]))
    elif (isinstance(e, Multiplication)):
        return reduce(lambda x, y: x | y, [__collectArrayRefs(se) for se in e.subexps], set([]))
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def __getLineNo(e):
    if (hasattr(e, 'line_no')):
        return e.line_no
    else:
        return None


class __SymbolTable:

    def __init__(self):
        self.range_tab = {}
        self.index_tab = {}
        self.array_tab = {}
        self.expand_tab = {}
        self.volatile_tab = {}
        self.assign_seq = []
        self.iteration = None


def symCheck(trans_unit):
    for ce in trans_unit.comp_elems:
        __symCheckCompElem(ce)


def __symCheckCompElem(comp_elem):
    for e in comp_elem.elems:
        if (isinstance(e, ast.absyn.AssignStmt)):
            __symCheckAssignStmt(e)


def __symCheckAssignStmt(s):
    global_sym_groups = s.lhs.sym_groups
    __symCheckExp(s.rhs, global_sym_groups)


def __symCheckExp(e, global_sym_groups):
    if (isinstance(e, ast.absyn.Array)):
        __symCheckArray(e, global_sym_groups)
    elif (isinstance(e, ast.absyn.Addition)):
        __symCheckAddition(e, global_sym_groups)
    elif (isinstance(e, ast.absyn.Multiplication)):
        __symCheckMultiplication(e, global_sym_groups)
    else:
        raise FrontEndError('%s: unknown declaration' % __name__)


def __symCheckArray(e, global_sym_groups):
    global_sets = map(set, global_sym_groups)
    this_sets = map(set, e.sym_groups)
    for s1 in global_sets:
        found = False
        for s2 in this_sets:
            if (s1 == s2):
                found = True
        if (not found):
            raise InputError(
                'contains an assignment where the permutation symmetry of array "%s" mismatches with LHS' % e.name
            )


def __symCheckAddition(e, global_sym_groups):
    pass


def __symCheckMultiplication(e, global_sym_groups):
    global_sets = map(set, global_sym_groups)
    this_sets = map(set, e.sym_groups)
    for s1 in global_sets:
        for s2 in this_sets:
            if (not ((s1 >= s2) or len(s1 & s2) == 0)):
                raise InputError('contains an assignment where permutation symmetry of LHS is not equal ' +
                                 'or not in higher order than RHS')

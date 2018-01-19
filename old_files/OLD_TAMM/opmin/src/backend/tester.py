from error import BackEndError
from ast.absyn import ArrayDecl, RangeDecl, IndexDecl, AssignStmt, Array, Addition, Multiplication


__RANGE_VALUE = 2

__PYTHON_EXE = '#! /usr/contrib/bin/python'
__MODULE_NAME = 'opmin_tester'
__FILE_NAME = __MODULE_NAME + '.py'
__VAR_NAME_PREFIX = '_temp_'
__ACCESSOR_FUNC_NAME = 'run'
__INPUT_ARR_SUFFIX = '_input'
__ORIG_ARR_SUFFIX = '_original'
__OPT_ARR_SUFFIX = '_optimum'


def test(orig_t_unit, opt_t_unit):
    is_correct = True
    for (orig_comp_elem, opt_comp_elem) in zip(orig_t_unit.comp_elems, opt_t_unit.comp_elems):
        is_correct = is_correct and __testCompElem(orig_comp_elem, opt_comp_elem)
    return is_correct


def __testCompElem(orig_comp_elem, opt_comp_elem):

    # generate testing code
    code = __generateTestingCode(orig_comp_elem, opt_comp_elem)

    # write to testing file
    try:
        f = open(__FILE_NAME, 'w')
        f.write(code)
        f.close()
    except:
        print 'error: cannot generate testing files: %s' % __FILE_NAME
        print 'warning: testing phase is skipped'

    # import testing module
    mod = __import__(__MODULE_NAME)

    # run tester
    func = getattr(mod, __ACCESSOR_FUNC_NAME)
    is_correct = func()

    # delete the testing file
    # os.unlink(__FILE_NAME)

    # return result
    return is_correct


def __generateTestingCode(orig_comp_elem, opt_comp_elem):
    (input_arrs_tab, result_arrs_tab, range_tab) = __collectDeclInfo(orig_comp_elem)
    vname_generator = __VarNameGenerator(__VAR_NAME_PREFIX)

    code = __generateHeader()
    code += '\n #---------- Declarations ---------- \n\n'
    code += __generateDecl(input_arrs_tab, range_tab)
    code += '\n #---------- Original Equation ---------- \n'
    code += __generateEq(orig_comp_elem, vname_generator, input_arrs_tab)
    code += __generateResultStoring(result_arrs_tab, __ORIG_ARR_SUFFIX)
    code += '\n #---------- Optimized Equation ---------- \n'
    code += __generateEq(opt_comp_elem, vname_generator, input_arrs_tab)
    code += __generateResultStoring(result_arrs_tab, __OPT_ARR_SUFFIX)
    code += '\n #---------- Results Comparison ---------- \n\n'
    code += __generateComparisons(result_arrs_tab, __ORIG_ARR_SUFFIX, __OPT_ARR_SUFFIX)
    code += '\nprint run()'
    return code


def __generateHeader():
    code = ''
    code += __PYTHON_EXE + '\n'
    code += 'import random' + '\n'
    code += 'from helper import makeSymArray, makeArray, replicateArray, replicateSymArray'
    code += '\n'
    code += 'def ' + __ACCESSOR_FUNC_NAME + '(): \n'
    return code


def __generateDecl(input_arrs_tab, range_tab):
    code = ''
    for (r, v) in range_tab.iteritems():
        code += ' ' + str(r) + ' = ' + str(v) + '\n'
    for (a, rs) in input_arrs_tab.iteritems():
        code += ' ' + str(a) + __INPUT_ARR_SUFFIX + ' = makeArray([' + ','.join(map(str, rs)) + '], True) \n'
    return code


def __generateEq(comp_elem, vname_generator, input_arrs_tab):
    code = '\n'
    index_tab = {}
    for e in comp_elem.elems:
        code += __generateElem(e, index_tab, vname_generator, input_arrs_tab)
    code = code.replace('\n', '\n ')
    return code


def __generateResultStoring(result_arrs_tab, suffix):
    code = '\n'
    for (a, r) in result_arrs_tab.iteritems():
        code += ' ' + str(a) + suffix + ' = replicateArray(' + str(a) + ') \n'
    return code


def __generateComparisons(result_arrs_tab, suffix1, suffix2):
    code = ''
    for (a, r) in result_arrs_tab.iteritems():
        code += ' print ' + str(a) + suffix1 + '\n'
        code += ' print ' + str(a) + suffix2 + '\n'
        code += ' if (' + str(a) + suffix1 + ' != ' + str(a) + suffix2 + '): \n'
        code += '  return False \n'
    code += ' return True \n'
    return code


def __generateElem(e, index_tab, vname_generator, input_arrs_tab):
    code = ''
    if (isinstance(e, IndexDecl)):
        index_tab[e.name] = e.range
    elif (isinstance(e, ArrayDecl)):
        if (e.name in input_arrs_tab):
            code += str(e.name) + ' = replicateSymArray(' + str(e.name) + __INPUT_ARR_SUFFIX
            code += ',['
            code += ','.join(e.upper_ranges + e.lower_ranges)
            code += '],'
            code += str(e.sym_groups)
            code += ') \n'
        else:
            code += str(e.name) + ' = makeArray(['
            code += ','.join(e.upper_ranges + e.lower_ranges)
            code += ']) \n'
    elif (isinstance(e, AssignStmt)):
        code += __generateAssignStmt(e, index_tab, vname_generator, input_arrs_tab)
    return code


def __generateAssignStmt(s, index_tab, vname_generator, input_arrs_tab):
    code_seq = []

    loop_inames = s.lhs.upper_inds + s.lhs.lower_inds
    lhs = __generateExp(s.lhs, index_tab, vname_generator, input_arrs_tab, code_seq)
    rhs = __generateExp(s.rhs, index_tab, vname_generator, input_arrs_tab, code_seq)
    body = lhs + ' = ' + rhs

    code = ''
    code += reduce(lambda x, y: x + y, code_seq, '')
    code += __generateLoop(loop_inames, body, index_tab)
    return code


def __generateExp(e, index_tab, vname_generator, input_arrs_tab, code_seq):
    code = ''
    if (isinstance(e, Array)):
        if (e.coef == -1):
            code += '-'
        elif (e.coef != -1 and e.coef != 1):
            code += '(' + str(e.coef) + ' * '
        code += str(e.name)
        if (len(e.upper_inds + e.lower_inds) > 0):
            code += '['
            code += ']['.join(e.upper_inds + e.lower_inds)
            code += ']'
        if (e.coef != -1 and e.coef != 1):
            code += ')'
    elif (isinstance(e, Addition)):
        if (e.coef != 1):
            raise BackEndError('%s: coefficient of an addition must be 1' % __name__)

        upper_ranges = map(lambda x: index_tab[x], e.upper_inds)
        lower_ranges = map(lambda x: index_tab[x], e.lower_inds)
        tmp_arr_decl = ArrayDecl(vname_generator.generate(), upper_ranges, lower_ranges, [], [])

        tmp_arr_ref = Array(tmp_arr_decl.name, 1, e.upper_inds, e.lower_inds, [])
        lhs = __generateExp(tmp_arr_ref, index_tab, vname_generator, input_arrs_tab, code_seq)
        rhs = ' + '.join(map(
            lambda x: '(' + __generateExp(x, index_tab, vname_generator, input_arrs_tab, code_seq) + ')', e.subexps)
        )
        body = lhs + ' += ' + rhs

        code_seq.append(__generateElem(tmp_arr_decl, index_tab, vname_generator, input_arrs_tab))
        code_seq.append(__generateLoop(e.upper_inds + e.lower_inds, body, index_tab))
        code += lhs
    elif (isinstance(e, Multiplication)):
        upper_ranges = map(lambda x: index_tab[x], e.upper_inds)
        lower_ranges = map(lambda x: index_tab[x], e.lower_inds)
        tmp_arr_decl = ArrayDecl(vname_generator.generate(), upper_ranges, lower_ranges, [], [])

        tmp_arr_ref = Array(tmp_arr_decl.name, 1, e.upper_inds, e.lower_inds, [])
        lhs = __generateExp(tmp_arr_ref, index_tab, vname_generator, input_arrs_tab, code_seq)
        rhs = ' * '.join(map(
            lambda x: '(' + __generateExp(x, index_tab, vname_generator, input_arrs_tab, code_seq) + ')', e.subexps)
        )
        if (e.coef == -1):
            rhs = '-(' + rhs + ')'
        elif (e.coef != -1 and e.coef != 1):
            rhs = '(' + str(e.coef) + ' * ' + rhs + ')'
        body = lhs + ' += ' + rhs

        code_seq.append(__generateElem(tmp_arr_decl, index_tab, vname_generator, input_arrs_tab))
        code_seq.append(__generateLoop(e.upper_inds + e.lower_inds + e.sum_inds, body, index_tab))
        code += lhs
    else:
        raise BackEndError('%s: unknown expression' % __name__)
    return code


def __generateLoop(loop_inames, body, index_tab):
    code = ''
    indent = ''
    for iname in loop_inames:
        code += indent + 'for ' + str(iname) + ' in range(0, ' + str(index_tab[iname]) + '):' + '\n'
        indent += ' '
    code += indent + body + '\n'
    return code


__DECLARED = 0
__REFERENCED = 1
__ASSIGNED = 2


def __collectDeclInfo(ce):
    input_arrs_tab = {}
    result_arrs_tab = {}
    range_tab = {}
    for e in ce.elems:
        __collectElem(e, input_arrs_tab, result_arrs_tab, range_tab)

    tmp = {}
    for (k, v) in input_arrs_tab.iteritems():
        if (v[0] == __REFERENCED):
            tmp[k] = v[1]
    input_arrs_tab = tmp

    tmp = {}
    for (k, v) in result_arrs_tab.iteritems():
        if (v[0] == __ASSIGNED):
            tmp[k] = v[1]
    result_arrs_tab = tmp

    return (input_arrs_tab, result_arrs_tab, range_tab)


def __collectElem(e, input_arrs_tab, result_arrs_tab, range_tab):
    if (isinstance(e, RangeDecl)):
        range_tab[e.name] = __RANGE_VALUE
    elif (isinstance(e, ArrayDecl)):
        ranges = e.upper_ranges + e.lower_ranges
        input_arrs_tab[e.name] = [__DECLARED, ranges]
        result_arrs_tab[e.name] = [__DECLARED, ranges]
    elif (isinstance(e, AssignStmt)):
        __collectExp(e.rhs, input_arrs_tab, result_arrs_tab)
        if (input_arrs_tab[e.lhs.name][0] == __DECLARED):
            input_arrs_tab[e.lhs.name][0] = __ASSIGNED
        result_arrs_tab[e.lhs.name][0] = __ASSIGNED


def __collectExp(e, input_arrs_tab, result_arrs_tab):
    if (isinstance(e, Array)):
        if (input_arrs_tab[e.name][0] == __DECLARED):
            input_arrs_tab[e.name][0] = __REFERENCED
        result_arrs_tab[e.name][0] = __REFERENCED
    elif (isinstance(e, Addition) or isinstance(e, Multiplication)):
        for se in e.subexps:
            __collectExp(se, input_arrs_tab, result_arrs_tab)
    else:
        raise BackEndError('%s: unknown expression' % __name__)


class __VarNameGenerator:

    def __init__(self, prefix):
        self.num = 1
        self.prefix = prefix

    def generate(self):
        iname = self.prefix + str(self.num)
        self.num += 1
        return iname

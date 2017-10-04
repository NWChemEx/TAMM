from error import FrontEndError
from absyn import (
    Decl,
    Stmt,
    AssignStmt,
    Parenth,
    NumConst,
    Array,
    Addition,
    Multiplication
)


def fold(trans_unit):
    for ce in trans_unit.comp_elems:
        __foldCompElem(ce)


def __foldCompElem(comp_elem):
    for e in comp_elem.elems:
        __foldElem(e)


def __foldElem(e):
    if (isinstance(e, Decl)):
        pass
    elif (isinstance(e, Stmt)):
        __foldStmt(e)
    else:
        raise FrontEndError('%s: unknown element' % __name__)


def __foldStmt(s):
    if (isinstance(s, AssignStmt)):
        return __foldAssignStmt(s)
    else:
        raise FrontEndError('%s: unknown statement' % __name__)


def __foldAssignStmt(s):
    s.rhs = __foldExp(s.rhs)


def __foldExp(e):
    if (isinstance(e, Parenth)):
        raise FrontEndError('%s: parenthesized expression must not exist during constant-folding process' % __name__)
    elif (isinstance(e, NumConst)):
        return __foldNumConst(e)
    elif (isinstance(e, Array)):
        return __foldArray(e)
    elif (isinstance(e, Addition)):
        return __foldAddition(e)
    elif (isinstance(e, Multiplication)):
        return __foldMultiplication(e)
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def __foldNumConst(e):
    return e


def __foldArray(e):
    return e


def __foldAddition(e):
    e.subexps = [__foldExp(se) for se in e.subexps]

    const = 0
    non_consts = []
    for se in e.subexps:
        if (isinstance(se, NumConst)):
            const += se.value * se.coef
        else:
            non_consts.append(se)

    if (len(non_consts) == 0):
        return __makeNumConst(const)
    elif (len(non_consts) == 1):
        if (const == 0):
            return non_consts[0]
        else:
            e.subexps = [__makeNumConst(const)] + non_consts
            return e
    else:
        if (const == 0):
            e.subexps = non_consts
            return e
        else:
            e.subexps = [__makeNumConst(const)] + non_consts
            return e


def __foldMultiplication(e):
    e.subexps = [__foldExp(se) for se in e.subexps]

    const = e.coef
    e.setCoef(1)
    non_consts = []
    for se in e.subexps:
        if (isinstance(se, NumConst)):
            const *= se.value * se.coef
        else:
            const *= se.coef
            se.setCoef(1)
            non_consts.append(se)

    if (len(non_consts) == 0):
        return __makeNumConst(const)
    elif (len(non_consts) == 1):
        if (const == 0):
            return NumConst(0)
        else:
            non_consts[0].multiplyCoef(const)
            return non_consts[0]
    else:
        if (const == 0):
            return NumConst(0)
        else:
            e.multiplyCoef(const)
            e.subexps = non_consts
            return e


def __makeNumConst(value):
    if (value < 0):
        n = NumConst(-value)
        n.inverseSign()
    else:
        n = NumConst(value)
    return n

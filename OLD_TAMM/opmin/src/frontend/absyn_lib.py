from error import FrontEndError
from absyn import Identifier, Parenth, NumConst, Array, Addition, Multiplication


def getIndices(e):
    if (isinstance(e, Parenth)):
        return getIndices(e.exp)
    elif (isinstance(e, NumConst)):
        return []
    elif (isinstance(e, Array)):
        return [i.replicate() for i in e.inds]
    elif (isinstance(e, Addition)):
        return getIndices(e.subexps[0])
    elif (isinstance(e, Multiplication)):
        inames = reduce(lambda x,y: x+y, [map(lambda x: x.name, getIndices(se)) for se in e.subexps], [])
        #print inames
        external_inames = []
        for i in inames:
            if (inames.count(i) == 1):
                external_inames.append(i)
        print external_inames
        return [Identifier(i) for i in external_inames]
    else:
        raise FrontEndError('%s: unknown expression' % __name__)


def getSumIndices(e):
    if (isinstance(e, Multiplication)):
        inames = reduce(lambda x,y: x+y, [map(lambda x: x.name, getIndices(se)) for se in e.subexps], [])
        sum_inames = []
        for i in inames:
            if (inames.count(i) > 1 and i not in sum_inames):
                sum_inames.append(i)
        return [Identifier(i) for i in sum_inames]
    else:
        raise FrontEndError('%s: illegal expression type' % __name__)



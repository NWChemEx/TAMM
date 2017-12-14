# Absyn
#  |
#  +-- TranslationUnit
#  |
#  +-- CompoundElem
#  |
#  +-- Elem
#  |    |
#  |    +-- Decl
#  |    |    |
#  |    |    +-- RangeDecl
#  |    |    +-- IndexDecl
#  |    |    +-- ArrayDecl
#  |    |    +-- ExpandDecl
#  |    |    +-- VolatileDecl
#  |    |    +-- IterationDecl
#  |    |
#  |    +-- Stmt
#  |         |
#  |         +-- AssignStmt
#  |
#  +-- Exp
#       |
#       +-- Array
#       +-- Multiplication
#       +-- Addition


class Absyn:

    def __init__(self):
        pass

    def replicate(self):
        raise NotImplementedError('%s: abstract function "replicate" unimplemented' % (self.__class__.__name__))

    def __repr__(self):
        raise NotImplementedError('%s: abstract function "__repr__" unimplemented' % (self.__class__.__name__))

    def __str__(self):
        return repr(self)


class TranslationUnit(Absyn):

    def __init__(self, comp_elems):
        Absyn.__init__(self)
        self.comp_elems = comp_elems

    def replicate(self):
        r_comp_elems = [ce.replicate() for ce in self.comp_elems]
        return TranslationUnit(r_comp_elems)

    def __repr__(self):
        return '\n'.join(map(lambda x: '\n' + str(x), self.comp_elems))


class CompoundElem(Absyn):

    def __init__(self, elems):
        Absyn.__init__(self)
        self.elems = elems

    def replicate(self):
        r_elems = [e.replicate() for e in self.elems]
        return CompoundElem(r_elems)

    def remove(self, index):
        del self.elems[index]

    def getAllInterNames(self):
        inter_names = []

        for e in self.elems:
            if (isinstance(e, AssignStmt)):
                inter_names.append(e.lhs.name)
            else:
                pass
        else:
            pass

        inter_names.sort(lambda x, y: compareInterNames(x, y))

        return inter_names

    def reArrange(self):
        while True:
            decl = []
            inter = []
            for e in self.elems:
                if isinstance(e, Stmt):
                    inter.append(e)
                else:
                    decl.append(e)

            poplist = []
            calculated = []
            check = True
            for e in inter:
                calculated.append(e.lhs.name)
                for sub in e.rhs.subexps:
                    if not sub.name.startswith('_a'):
                        calculated.append(sub.name)
                    if sub.name not in calculated:
                        poplist.append(sub.name)
                        check = False
                        break

            inter2 = []
            temp = []
            for x in range(0, len(inter)):
                if inter[x].lhs.name in poplist:
                    temp.append(inter[x])
                else:
                    inter2.append(inter[x])

            self.elems = decl + temp + inter2
            if check:
                break

        return False

    def removeRedundant(self):
        inter_names = self.getAllInterNames()
        removeList = []
        for s in inter_names:
            used = False
            for e in self.elems:
                if isinstance(e, AssignStmt):  # intermediates
                    if (isinstance(e.rhs, Array)):
                        pass
                    else:
                        for sub in e.rhs.subexps:
                            if s == sub.name:
                                used = True
                                break
            if not used and s.startswith('_a'):
                removeList.append(s)

        self.removeElems(removeList)
        self.removeDuplicate()

    def removeDuplicate(self):
        new_elem = []
        exist = []
        for e in self.elems:
            if isinstance(e, AssignStmt):  # intermediates
                if e.lhs.name in exist:
                    continue
                else:
                    new_elem.append(e)
                    exist.append(e.lhs.name)
            else:
                new_elem.append(e)

        self.elems = new_elem

    def checkIfUsed(self, i_name):
        used = False
        for e in self.elems:
            if isinstance(e, AssignStmt):  # intermediates
                if (isinstance(e.rhs, Array)):
                    pass
                else:
                    for sub in e.rhs.subexps:
                        if i_name == sub.name:
                            used = True
                            break
        return used

    def removeElems(self, removeList):
        new_elems = []
        for e in self.elems:
            if (isinstance(e, AssignStmt)):
                if (isinstance(e.rhs, Array)):
                    pass
                else:
                    if e.lhs.name in removeList:
                        continue
            elif (isinstance(e, ArrayDecl)):
                if e.name in removeList:
                    continue
            else:
                pass
            new_elems.append(e)
        self.elems = new_elems

    def checkAddition(self):
        decl = []
        inter = []
        for e in self.elems:
            if isinstance(e, Stmt):
                inter.append(e)
            else:
                decl.append(e)

        for e in inter:
            if isinstance(e.rhs, Addition):
                lhs_inds = e.lhs.upper_inds + e.lhs.lower_inds
                rhs1_inds = e.rhs.subexps[0].upper_inds + e.rhs.subexps[0].lower_inds
                if lhs_inds != rhs1_inds:
                    self.modifyAddition(e.lhs.name)

    def checkMultiplication(self):
        decl = []
        inter = []
        for e in self.elems:
            if isinstance(e, Stmt):
                inter.append(e)
            else:
                decl.append(e)

        for e in inter:
            if isinstance(e.rhs, Multiplication):
                lhs_inds = e.lhs.upper_inds + e.lhs.lower_inds
                new_mult = Multiplication(e.rhs.subexps)
                rhs_inds = new_mult.upper_inds + new_mult.lower_inds

                if lhs_inds != rhs_inds:
                    self.modifyMultiplication(e.lhs.name)

    def modifyMultiplication(self, iname):
        # modify iname in lhs
        upc = []  # upper code
        lwc = []  # lower code

        for e in self.elems:
            if (isinstance(e, AssignStmt)):
                if (isinstance(e.rhs, Multiplication)):
                    if e.lhs.name == iname:
                        new_mult = Multiplication(e.rhs.subexps)
                        # rhs_inds = new_mult.upper_inds + new_mult.lower_inds
                        [upc, lwc] = calculateCode(
                            e.lhs.upper_inds,
                            e.lhs.lower_inds,
                            new_mult.upper_inds,
                            new_mult.lower_inds,
                        )
                        e.lhs.upper_inds = new_mult.upper_inds
                        e.lhs.lower_inds = new_mult.lower_inds

        if upc == [] or lwc == []:
            return

        # modify iname in rhs
        for e in self.elems:
            if (isinstance(e, AssignStmt)):
                if (isinstance(e.rhs, Addition)):
                    for sub in e.rhs.subexps:
                        if sub.name == iname:
                            [sub.upper_inds, sub.lower_inds] = renameByCode(
                                sub.upper_inds,
                                sub.lower_inds,
                                upc,
                                lwc,
                            )
                if (isinstance(e.rhs, Multiplication)):
                    for sub in e.rhs.subexps:
                        if sub.name == iname:
                            [sub.upper_inds, sub.lower_inds] = renameByCode(
                                sub.upper_inds,
                                sub.lower_inds,
                                upc,
                                lwc,
                            )

    def modifyAddition(self, iname):  # sort name of an Addition
        # modify iname in lhs
        upc = []  # upper code
        lwc = []  # lower code

        for e in self.elems:
            if (isinstance(e, AssignStmt)):
                if (isinstance(e.rhs, Addition)):
                    if e.lhs.name == iname:
                        temp = e.rhs.subexps[0]
                        e.rhs.subexps[0] = e.rhs.subexps[1]
                        e.rhs.subexps[1] = temp

                        [upc, lwc] = calculateCode(
                            e.lhs.upper_inds,
                            e.lhs.lower_inds,
                            e.rhs.subexps[0].upper_inds,
                            e.rhs.subexps[0].lower_inds,
                        )
                        e.lhs.upper_inds = e.rhs.subexps[0].upper_inds
                        e.lhs.lower_inds = e.rhs.subexps[0].lower_inds

        if upc == [] or lwc == []:
            return

        # modify iname in rhs
        for e in self.elems:
            if (isinstance(e, AssignStmt)):
                if (isinstance(e.rhs, Addition)):
                    for sub in e.rhs.subexps:
                        if sub.name == iname:
                            [sub.upper_inds, sub.lower_inds] = renameByCode(sub.upper_inds, sub.lower_inds, upc, lwc)
                if (isinstance(e.rhs, Multiplication)):
                    for sub in e.rhs.subexps:
                        if sub.name == iname:
                            [sub.upper_inds, sub.lower_inds] = renameByCode(sub.upper_inds, sub.lower_inds, upc, lwc)

    def __repr__(self):
        s = '{'
        s += ''.join(map(lambda x: '\n  ' + str(x), self.elems))
        if (len(self.elems) > 0):
            s += '\n'
        s += '}'
        return s


class Elem(Absyn):

    def __init__(self):
        Absyn.__init__(self)


class Decl(Elem):

    def __init__(self):
        Elem.__init__(self)


class RangeDecl(Decl):

    def __init__(self, name, value):
        Decl.__init__(self)
        self.name = name
        self.value = value

    def replicate(self):
        return RangeDecl(self.name, self.value)

    def __repr__(self):
        s = ''
        s += 'range ' + str(self.name) + ' = ' + str(self.value) + ';'
        return s


class IndexDecl(Decl):

    def __init__(self, name, range):
        Decl.__init__(self)
        self.name = name
        self.range = range

    def replicate(self):
        return IndexDecl(self.name, self.range)

    def __repr__(self):
        s = ''
        s += 'index ' + str(self.name) + ' = ' + str(self.range) + ';'
        return s


class ArrayDecl(Decl):

    def __init__(self, name, upper_ranges, lower_ranges, sym_groups, vsym_groups):
        Decl.__init__(self)
        self.name = name
        self.upper_ranges = upper_ranges
        self.lower_ranges = lower_ranges
        self.sym_groups = sym_groups
        self.vsym_groups = vsym_groups

    def replicate(self):
        r_sym_groups = [g[:] for g in self.sym_groups]
        r_vsym_groups = [h[:] for h in self.vsym_groups]
        return ArrayDecl(self.name, self.upper_ranges[:], self.lower_ranges[:], r_sym_groups, r_vsym_groups)

    def __repr__(self):
        s = ''
        s += 'array ' + str(self.name) + '(['
        s += ','.join(map(str, self.upper_ranges))
        s += ']['
        s += ','.join(map(str, self.lower_ranges))
        s += ']:'
        for g in self.sym_groups:
            s += '(' + ','.join(map(lambda x: str(x), g)) + ')'
        s += ':'
        for g in self.vsym_groups:
            s += '<' + ','.join(map(lambda x: str(x), g)) + '>'
        s += ');'
        return s


class ExpandDecl(Decl):

    def __init__(self, arr):
        Decl.__init__(self)
        self.arr = arr

    def replicate(self):
        return ExpandDecl(self.arr)

    def __repr__(self):
        s = ''
        s += 'expand ' + str(self.arr) + ';'
        return s


class VolatileDecl(Decl):

    def __init__(self, arr):
        Decl.__init__(self)
        self.arr = arr

    def replicate(self):
        return VolatileDecl(self.arr)

    def __repr__(self):
        s = ''
        s += 'volatile ' + str(self.arr) + ';'
        return s


class IterationDecl(Decl):

    def __init__(self, value):
        Decl.__init__(self)
        self.value = value

    def replicate(self):
        return IterationDecl(self.value)

    def __repr__(self):
        s = ''
        s += 'iteration ' + str(self.value) + ';'
        return s


class Stmt(Elem):

    def __init__(self):
        Elem.__init__(self)


class AssignStmt(Stmt):

    def __init__(self, lhs, rhs):
        Stmt.__init__(self)
        self.lhs = lhs
        self.rhs = rhs

    def replicate(self):
        return AssignStmt(self.lhs.replicate(), self.rhs.replicate())

    def __repr__(self):
        s = ''
        s += self.lhs.strRepr('')
        s += ' = \n    '
        s += self.rhs.strRepr('      ')
        s += ';'
        return s


class Exp(Absyn):

    def __init__(self, coef=1, upper_inds=None, lower_inds=None, sym_groups=None, vsym_groups=None):
        Absyn.__init__(self)
        self.coef = coef
        self.upper_inds = upper_inds
        self.lower_inds = lower_inds
        self.sym_groups = sym_groups
        self.vsym_groups = vsym_groups

    def isNegative(self):
        return self.coef < 0

    def isPositive(self):
        return self.coef > 0

    def __repr__(self):
        return self.strRepr(' ')

    def strRepr(self, indent):
        raise NotImplementedError('%s: abstract function "strRepr" unimplemented' % (self.__class__.__name__))

    def setCoef(self, coef):
        raise NotImplementedError('%s: abstract function "setCoef" unimplemented' % (self.__class__.__name__))

    def multiplyCoef(self, coef):
        raise NotImplementedError('%s: abstract function "multiplyCoef" unimplemented' % (self.__class__.__name__))

    def inverseSign(self):
        raise NotImplementedError('%s: abstract function "inverseSign" unimplemented' % (self.__class__.__name__))


class Array(Exp):

    def __init__(self, name, coef=1, upper_inds=None, lower_inds=None, sym_groups=None, vsym_groups=None):
        Exp.__init__(self, coef, upper_inds, lower_inds, sym_groups, vsym_groups)
        self.name = name

    def replicate(self):
        r_sym_groups = [g[:] for g in self.sym_groups]
        r_vsym_groups = [g[:] for g in self.vsym_groups]
        return Array(self.name, self.coef, self.upper_inds[:], self.lower_inds[:], r_sym_groups, r_vsym_groups)

    def strRepr(self, indent):
        s = ''
        if (self.coef != 1):
            s += str(self.coef) + ' * '
        s += str(self.name)
        s += '[' + ','.join(map(str, self.upper_inds)) + '][' + ','.join(map(str, self.lower_inds)) + ']'
        s += ':{'
        for g in self.sym_groups:
            s += '(' + ','.join(g) + ')'
        s += '}'
        s += ':{'
        for g in self.vsym_groups:
            s += '<' + ','.join(g) + '>'
        s += '}'
        return s

    def setCoef(self, coef):
        self.coef = coef

    def multiplyCoef(self, coef):
        self.coef *= coef

    def inverseSign(self):
        self.multiplyCoef(-1)


class Addition(Exp):

    def __init__(self, subexps, coef=1, upper_inds=None, lower_inds=None, sym_groups=None, vsym_groups=None):
        if (upper_inds is None and lower_inds is None and sym_groups is None and vsym_groups is None):
            upper_inds = subexps[0].upper_inds[:]
            lower_inds = subexps[0].lower_inds[:]
            [sym_groups, vsym_groups] = self.__computeSymmetry(subexps)
        Exp.__init__(self, coef, upper_inds, lower_inds, sym_groups, vsym_groups)
        self.subexps = subexps

    def replicate(self):
        r_subexps = [se.replicate() for se in self.subexps]
        r_sym_groups = [g[:] for g in self.sym_groups]
        r_vsym_groups = [g[:] for g in self.vsym_groups]
        return Addition(r_subexps, self.coef, self.upper_inds[:], self.lower_inds[:], r_sym_groups, r_vsym_groups)

    def strRepr(self, indent):
        s = ''
        s += 'add('
        if (self.coef != 1):
            s += str(self.coef) + ', '
        s += '[' + ','.join(map(str, self.upper_inds)) + '][' + ','.join(map(str, self.lower_inds)) + ']'

        s += ':{'
        for g in self.sym_groups:
            s += '(' + ','.join(g) + ')'
        s += '}'

        s += ':{'
        for g in self.vsym_groups:
            s += '<' + ','.join(g) + '>'
        s += '}'

        s += ','
        s += ''.join(map(lambda x: '\n' + indent + x.strRepr(indent + '  '), self.subexps))
        s += ')'
        return s

    def setCoef(self, coef):
        for se in self.subexps:
            se.setCoef(coef)

    def multiplyCoef(self, coef):
        for se in self.subexps:
            se.multiplyCoef(coef)

    def inverseSign(self):
        for se in self.subexps:
            se.inverseSign()

    def __computeSymmetry(self, subexps):
        tensor1 = subexps[0]
        tensor2 = subexps[1]

        indices = tensor1.upper_inds + tensor1.lower_inds
        ind_set = set(indices)

        sym_groups = []
        for s1 in tensor1.sym_groups:
            for s2 in tensor2.sym_groups:
                if s1 == s2:
                    sym_groups.append(s1)

        ind_set2 = set()
        for x in sym_groups:
            ind_set2 |= set(x)

        singles = list(ind_set - ind_set2)
        for s in singles:
            sym_groups.append([s])

        vsym_groups = [[], []]  # [g[:] for g in subexps[0].vsym_groups]

        no_anti = True
        for s in sym_groups:
            if len(s) > 1:
                no_anti = False

        if no_anti is False:
            return [sym_groups, vsym_groups]

        if tensor1.sym_groups == tensor2.sym_groups:
            if tensor1.vsym_groups == tensor2.vsym_groups:
                return [tensor1.sym_groups, tensor1.vsym_groups]

        return [sym_groups, vsym_groups]


class Multiplication(Exp):

    def __init__(
        self,
        subexps,
        coef=1,
        upper_inds=None,
        lower_inds=None,
        sum_inds=None,
        sym_groups=None,
        sum_sym_groups=None,
        vsym_groups=None,
        sum_vsym_groups=None
    ):
        if (
            upper_inds is None and lower_inds is None and sum_inds is None and sym_groups is None and
            sum_sym_groups is None and vsym_groups is None and sum_vsym_groups is None
        ):
            coef *= self.__computeCoef(subexps)
            (upper_inds, lower_inds, sum_inds, sym_groups, sum_sym_groups) = self.__computeIndicesAndSymmetry(subexps)
            (vsym_groups, sum_vsym_groups, extra_sym_groups) = self.__computeVertexSymmetry(subexps)

        Exp.__init__(self, coef, upper_inds, lower_inds, sym_groups, vsym_groups)
        self.sum_inds = sum_inds
        self.sum_sym_groups = sum_sym_groups
        self.sum_vsym_groups = sum_vsym_groups
        self.subexps = subexps
        self.ops = []

    def replicate(self):
        r_subexps = [se.replicate() for se in self.subexps]
        r_sym_groups = [g[:] for g in self.sym_groups]
        r_sum_sym_groups = [g[:] for g in self.sum_sym_groups]
        r_vsym_groups = [g[:] for g in self.vsym_groups]
        r_sum_vsym_groups = [g[:] for g in self.sum_vsym_groups]
        m = Multiplication(r_subexps, self.coef, self.upper_inds[:], self.lower_inds[:], self.sum_inds[:],
                           r_sym_groups, r_sum_sym_groups,
                           r_vsym_groups, r_sum_vsym_groups)
        m.ops = [o.replicate() for o in self.ops]
        return m

    def strRepr(self, indent):
        s = ''
        s += 'mult(\n'
        if (self.coef != 1):
            s += indent + 'coef: ' + str(self.coef) + '\n'
        s += indent + 'inds: '
        s += '[' + ','.join(map(str, self.upper_inds)) + '][' + ','.join(map(str, self.lower_inds)) + ']'
        s += ':{'
        for g in self.sym_groups:
            s += '(' + ','.join(g) + ')'
        s += '}:{'
        for g in self.vsym_groups:
            s += '(' + ','.join(g) + ')'
        s += '}\n'
        s += indent + 'sum_inds: '
        s += '[' + ','.join(map(str, self.sum_inds)) + ']'
        s += ':{'
        for g in self.sum_sym_groups:
            s += '(' + ','.join(g) + ')'
        s += '}:{'
        for g in self.sum_vsym_groups:
            s += '(' + ','.join(g) + ')'
        s += '}\n' + indent + 'ops: '
        s += '[' + ','.join(map(str, self.ops)) + ']\n'
        s += indent + 'subexps:'
        s += ''.join(map(lambda x: '\n' + indent + indent + x.strRepr(indent + '  '), self.subexps))
        s += ')'
        return s

    def setCoef(self, coef):
        self.coef = coef

    def multiplyCoef(self, coef):
        self.coef *= coef

    def inverseSign(self):
        self.multiplyCoef(-1)

    def setOps(self, global_sym_groups):
        (ops, new_sym_groups) = self.__computeOpsAndSymmetry(global_sym_groups)
        self.ops = ops
        self.sym_groups = new_sym_groups

    def __computeCoef(self, subexps):
        coef = 1
        for se in subexps:
            if (se.coef != 1):
                coef *= se.coef
                se.setCoef(1)
        return coef

    def __computeIndicesAndSymmetry(self, subexps):
        [inds_list, up_inds_list, lo_inds_list, sym_groups_list] = zip(
            *[(se.upper_inds + se.lower_inds, se.upper_inds, se.lower_inds, se.sym_groups) for se in subexps]
        )
        inds = reduce(lambda x, y: x + y, inds_list, [])
        sum_inds = []
        for i in inds:
            if (inds.count(i) > 1 and i not in sum_inds):
                sum_inds.append(i)

        up_inds = reduce(lambda x, y: x + y, up_inds_list, [])
        lo_inds = reduce(lambda x, y: x + y, lo_inds_list, [])
        up_ext_inds = filter(lambda x: x not in sum_inds, up_inds)
        lo_ext_inds = filter(lambda x: x not in sum_inds, lo_inds)

        ext_sym_groups = []
        sum_sym_groups = []

        for sg in sym_groups_list:
            for g in sg:
                ext_g = filter(lambda x: x not in sum_inds, g)
                sum_g = filter(lambda x: x in sum_inds, g)
                if (len(ext_g) > 0):
                    ext_sym_groups.append(ext_g)
                if (len(sum_g) > 0):
                    sum_sym_groups.append(sum_g)

        sym_groups_sets = [set(g) for g in sum_sym_groups]

        temp2 = []
        for s in sum_sym_groups:
            if (len(s) > 1 and s not in temp2):
                temp2.append(s)

        sum_sym_groups = []
        while (len(sym_groups_sets) > 0):
            next_sym_groups_sets = []
            cur_s = sym_groups_sets.pop(0)
            for s in sym_groups_sets:
                common_s = cur_s & s
                if (len(common_s) == 0):
                    next_sym_groups_sets.append(s)
                else:
                    sum_sym_groups.append(common_s)
                    remaining_s = s - common_s
                    if (len(remaining_s) > 0):
                        next_sym_groups_sets.append(remaining_s)
                    cur_s = cur_s - common_s
            sym_groups_sets = next_sym_groups_sets

        sum_sym_groups = map(lambda x: list(x), sum_sym_groups)

        return (up_ext_inds, lo_ext_inds, sum_inds, ext_sym_groups, sum_sym_groups)

    def __computeVertexSymmetryForTwoTensors(self, tensor1, tensor2):
        new_vsym_group1 = [[], []]
        new_vsym_group2 = [[], []]
        new_vsym_groups = [[], []]
        inds1 = tensor1.upper_inds + tensor1.lower_inds
        inds2 = tensor2.upper_inds + tensor2.lower_inds
        sum_inds = filter(lambda x: x in inds2, inds1)
        bothV = False
        if not (tensor1.vsym_groups == [] or tensor2.vsym_groups == []):
            bothV = True
            (up1, lo1) = (tensor1.vsym_groups[0], tensor1.vsym_groups[1])
            (up2, lo2) = (tensor2.vsym_groups[0], tensor2.vsym_groups[1])
            for x in range(0, len(up1)):  # x=0,1
                if up1[x] in sum_inds and lo1[x] not in sum_inds:
                    if up1[x] in lo2:
                        new_vsym_group1[0].append(up2[x])
                        new_vsym_group1[1].append(lo1[x])
                if up1[x] not in sum_inds and lo1[x] in sum_inds:
                    if lo1[x] in up2:
                        new_vsym_group2[0].append(up1[x])
                        new_vsym_group2[1].append(lo2[x])

        if len(new_vsym_group1[0]) < 2 and len(new_vsym_group2[0]) < 2:
            new_vsym_groups = [[], []]
        elif len(new_vsym_group1[0]) > len(new_vsym_group2[0]):
            new_vsym_groups = new_vsym_group1
        else:
            new_vsym_groups = new_vsym_group2

        extra_sym_group = []

        if not bothV:
            return [extra_sym_group, new_vsym_groups]

        return [extra_sym_group, new_vsym_groups]

    def __computeVertexSymmetry(self, subexps):
        sum_vsym_groups = []

        [extra_sym_groups, vsym_groups] = self.__computeVertexSymmetryForTwoTensors(subexps[0], subexps[1])
        return (vsym_groups, sum_vsym_groups, extra_sym_groups)

    def __computeOpsAndSymmetry(self, global_sym_groups):
        tab = {}

        for g in self.sym_groups:
            for i in g:
                tab[i] = g

        ops = []
        for g in global_sym_groups:
            gs = []
            for i in g:
                if (tab.has_key(i)):
                    cur_g = tab[i]
                    if (cur_g not in gs):
                        gs.append(cur_g[:])
            if (len(gs) > 1):
                ops.append(SymmetrizationOp(gs))

        new_sym_groups = []
        seen_inds = []
        for o in ops:
            g = reduce(lambda x, y: x + y, o.sym_groups, [])
            new_sym_groups.append(g)
            seen_inds.extend(g)
        for (i, g) in tab.iteritems():
            if (i not in seen_inds):
                new_sym_groups.append(g[:])
                seen_inds.extend(g)
        return (ops, new_sym_groups)


class SymmetrizationOp:

    def __init__(self, sym_groups):
        self.sym_groups = sym_groups

    def replicate(self):
        r_sym_groups = [g[:] for g in self.sym_groups]
        return SymmetrizationOp(r_sym_groups)

    def __repr__(self):
        s = ''
        s += 'S('
        for i, g in enumerate(self.sym_groups):
            if (i > 0):
                s += '|'
            s += ','.join(g)
        s += ')'
        return s

    def __str__(self):
        return repr(self)


def compareInterNames(n1, n2):
    if n1[0] != '_':
        return 1
    if n2[0] != '_':
        return -1

    num1 = int(n1[2:])
    num2 = int(n2[2:])

    if num1 > num2:
        return 1
    else:
        return -1


def calculateCode(from_upper_inds, from_lower_inds, to_upper_inds, to_lower_inds):
    upper_code = []
    lower_code = []

    for i in from_upper_inds:
        upper_code.append(to_upper_inds.index(i))
    for i in from_lower_inds:
        lower_code.append(to_lower_inds.index(i))

    return [upper_code, lower_code]


def renameByCode(upper_inds, lower_inds, upper_code, lower_code):
    new_upper_inds = []
    new_lower_inds = []

    for c in upper_code:
        new_upper_inds.append(upper_inds[c])
    for c in lower_code:
        new_lower_inds.append(lower_inds[c])
    return [new_upper_inds, new_lower_inds]

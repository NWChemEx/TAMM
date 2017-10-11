from error import OptimizerError
from ast.absyn import Array, Addition, Multiplication
from ast.absyn_lib import renameIndices, buildRenamingTable
from cost_model import countOpCost
from canon import canonicalizeExp


class ResultInfo:
    def __init__(self, lhs, rhs, from_inds, to_inds, i_arr_name_set, op_cost, is_volatile):
        self.lhs = lhs
        self.rhs = rhs
        self.from_inds = from_inds
        self.to_inds = to_inds
        self.i_arr_name_set = i_arr_name_set
        self.op_cost = op_cost
        self.is_volatile = is_volatile


class StmtTable:

    IARRAY_PREFIX = '_a'

    def __init__(self, use_common_subexp=True):
        self.use_common_subexp = use_common_subexp
        self.can_map = {}
        self.arr_map = {}

    def __str__(self):
        s = ''
        s += '\n' + 'can_map = \n'
        s += str(self.can_map)
        s += '\n' + 'arr_map = \n'
        s += str(self.arr_map)
        return s

    def __repr__(self):
        return str(self)

    def shallow_copy(self):
        tab = StmtTable(self.use_common_subexp)
        tab.can_map = self.can_map.copy()
        tab.arr_map = self.arr_map.copy()
        return tab

    def turnOnCommonSubexp(self):
        self.use_common_subexp = True

    def turnOffCommonSubexp(self):
        self.use_common_subexp = False

    def getAllIntermediateNames(self):
        inter_names = self.arr_map.keys()
        inter_names.sort(lambda x, y: compareInterNames(x, y))
        return inter_names

    def getAllIntermediateExps(self):
        exps = []
        for k in self.arr_map.keys():
            exp = self.arr_map[k].rhs
            exps.append(exp)
        return exps

    def getAllIntermediateCosts(self):
        costs = []
        for k in self.arr_map.keys():
            cost = self.arr_map[k].op_cost
            costs.append(cost)
        return costs

    def getAllIntermediateCanonicalForms(self):
        cfs = []
        for k in self.arr_map.keys():
            exp = self.arr_map[k].rhs
            can_forms_info = canonicalizeExp(exp)
            (cf, to_inds) = can_forms_info[0]
            cfs.append(cf)
        return cfs

    def getVolatileIntermediateNames(self, arr_names):
        return filter(lambda x: self.arr_map[x].is_volatile, arr_names)

    def getOrderedInfos(self, leading_arr_names, arr_names):
        sequence = []
        arr_names = list(arr_names)
        while (len(arr_names) > 0):
            leading = None
            for a in arr_names:
                info = self.arr_map[a]
                all_ahead = True
                for i in info.i_arr_name_set:
                    all_ahead = all_ahead and ((i in sequence) or (i in leading_arr_names))
                if (all_ahead):
                    leading = a
                    break
            assert(leading is not None), '%s: a leading array must exist' % __name__
            sequence.append(leading)
            del arr_names[arr_names.index(leading)]
        return map(lambda x: self.arr_map[x], sequence)

    def getCostForIntermediateNames(self, i_arr_names):
        cost = 0
        for a in i_arr_names:
            cost += self.arr_map[a].op_cost
        return cost

    def getRhs(self, arr):
        if (self.arr_map.has_key(arr.name)):
            info = self.arr_map[arr.name]
            rhs = info.rhs.replicate()
            from_inds = info.lhs.upper_inds + info.lhs.lower_inds
            to_inds = arr.upper_inds + arr.lower_inds
            ren_tab = buildRenamingTable(from_inds, to_inds)
            renameIndices(rhs, ren_tab)
            rhs.multiplyCoef(arr.coef)
            return rhs
        else:
            return None

    def getCf(self, arr_name):
        if (self.arr_map.has_key(arr_name)):
            exp = self.arr_map[arr_name].rhs
            can_forms_info = canonicalizeExp(exp)
            return can_forms_info
        else:
            return None

    def getCost(self, arr_name):
        if self.arr_map.has_key(arr_name):
            return self.arr_map[arr_name].op_cost
        else:
            return None

    def updateSame(self, a1, a2):  # change a2 to a1
        if self.arr_map.has_key(a1.name) and self.arr_map.has_key(a2.name):
            for k in self.arr_map.keys():
                this = self.arr_map[k]
                if this.lhs.name == a2.name:
                    del self.arr_map[this.lhs.name]
                else:
                    for x in range(0, len(this.rhs.subexps)):
                        if a2.name == this.rhs.subexps[x].name:
                            self.arr_map[k].rhs.subexps[x].name = a1.name

                            new_set = set([])
                            new_set.add(a1.name)
                            for i in self.arr_map[k].i_arr_name_set:
                                if i != a2.name:
                                    new_set.add(i)
                            self.arr_map[k].i_arr_name_set = new_set
        else:
            pass

    def removeRedundant(self, removeList):
        for i in self.getAllIntermediateNames():
            if i in removeList:
                del self.arr_map[i]
        return None

        inter_names = self.getAllIntermediateNames()

        while len(inter_names) != 0:
            this = inter_names.pop(0)
            used = False

            for n in inter_names:
                for s in self.arr_map[n].rhs.subexps:
                    if this == s.name:
                        used = True

            if not used:
                del self.arr_map[this]

    def getInfoForExp(self, exp, index_tab, volatile_tab, iteration):
        can_forms_info = canonicalizeExp(exp)

        (cf, to_inds) = can_forms_info[0]

        if (self.can_map.has_key(cf)):
            info = self.can_map[cf]
        else:
            (can_forms, from_maps) = zip(*can_forms_info)
            info = __insertInfoForExp(
                self, self, exp, list(can_forms), list(from_maps), index_tab, volatile_tab, iteration)

        from_inds = info.from_maps[info.can_forms.index(cf)]
        return ResultInfo(info.lhs, info.rhs, from_inds, to_inds, info.i_arr_name_set, info.op_cost, info.is_volatile)

    def updateWithStmtTab(self, stmt_tab, i_arr_name_set):
        for a in i_arr_name_set:
            if (not self.arr_map.has_key(a)):
                info = stmt_tab.arr_map[a]
                if (self.use_common_subexp):
                    for cf in info.can_forms:
                        self.can_map[cf] = info
                self.arr_map[a] = info

    __counter = 1
    global __EntryInfo, __generateNewArrayName, __getIntmSetAndVolatileList, __insertInfoForExp

    class __EntryInfo:
        def __init__(self, lhs, rhs, can_forms, from_maps, i_arr_name_set, op_cost, is_volatile):
            self.lhs = lhs
            self.rhs = rhs
            self.can_forms = can_forms
            self.from_maps = from_maps
            self.i_arr_name_set = i_arr_name_set
            self.op_cost = op_cost
            self.is_volatile = is_volatile

        def __str__(self):
            s = ''
            s += '\n' + 'lhs= ' + str(self.lhs)
            s += '\n' + 'rhs= ' + str(self.rhs)
            s += '\n' + 'i_arr_name_set= ' + str(self.i_arr_name_set)
            s += '\n' + 'op_cost= ' + str(self.op_cost)
            s += '\n' + 'is_volatile= ' + str(self.is_volatile)
            s += '\n'
            return s

        def __repr__(self):
            return str(self)

    def __generateNewArrayName(self):
        name = StmtTable.IARRAY_PREFIX + str(StmtTable.__counter)
        StmtTable.__counter += 1
        return name

    def __getIntmSetAndVolatileList(self, arr_names, stmt_tab):
        i_arr_name_set = set([])
        volatile_list = []
        for a in set(arr_names):
            if (stmt_tab.arr_map.has_key(a)):
                info = stmt_tab.arr_map[a]
                i_arr_name_set |= set([a]) | info.i_arr_name_set
                if (info.is_volatile):
                    volatile_list.append(a)
        return (i_arr_name_set, volatile_list)

    def __insertInfoForExp(self, stmt_tab, e, can_forms, from_maps, index_tab, volatile_tab, iteration):
        rhs = e.replicate()
        lhs = Array(
            __generateNewArrayName(self),
            1,
            rhs.upper_inds[:],
            rhs.lower_inds[:],
            [g[:] for g in rhs.sym_groups], [g[:] for g in rhs.vsym_groups]
        )  # array = tensor

        if (isinstance(e, Array)):
            (i_arr_name_set, volatile_list) = __getIntmSetAndVolatileList([e.name], stmt_tab)
        elif (isinstance(e, Addition) or isinstance(e, Multiplication)):
            (i_arr_name_set, volatile_list) = __getIntmSetAndVolatileList(
                self, [se.name for se in e.subexps], stmt_tab)
        else:
            raise OptimizerError('%s: unknown expression' % __name__)

        updated_volatile_tab = volatile_tab.copy()
        for v in volatile_list:
            updated_volatile_tab[v] = None

        (op_cost, is_volatile) = countOpCost(rhs, index_tab, updated_volatile_tab, iteration)

        info = __EntryInfo(lhs, rhs, can_forms, from_maps, i_arr_name_set, op_cost, is_volatile)

        if (stmt_tab.use_common_subexp):
            for cf in can_forms:
                stmt_tab.can_map[cf] = info
        stmt_tab.arr_map[lhs.name] = info

        return info


class UsageTable:

    def __init__(self, stmt_tab, top_arefs):
        self.stmt_tab = stmt_tab
        self.top_arefs = top_arefs

    def shallow_copy(self):
        return UsageTable(self.stmt_tab.shallow_copy(), self.top_arefs[:])

    def insertInfos(self, infos):
        for i in infos:
            if (not self.stmt_tab.arr_map.has_key(i.lhs.name)):
                if (self.stmt_tab.use_common_subexp):
                    for cf in i.can_forms:
                        self.stmt_tab.can_map[cf] = i
                self.stmt_tab.arr_map[i.lhs.name] = i

    def removeIntermediates(self, i_arrs, mults):
        def __makeRefMap(stmt_tab, top_arefs):
            all_arefs = top_arefs[:]
            for (a, i) in stmt_tab.arr_map.iteritems():
                all_arefs.extend(getARefNames(i.rhs))

            ref_map = {}
            for a in all_arefs:
                if (stmt_tab.arr_map.has_key(a)):
                    if (ref_map.has_key(a)):
                        ref_map[a] += 1
                    else:
                        ref_map[a] = 1
            return ref_map

        def __remove(stmt_tab, ref_map, aref):
            if (stmt_tab.arr_map.has_key(aref)):
                if (ref_map[aref] == 1):
                    info = stmt_tab.arr_map[aref]
                    if (stmt_tab.use_common_subexp):
                        for cf in info.can_forms:
                            del stmt_tab.can_map[cf]
                    del stmt_tab.arr_map[aref]
                    del ref_map[aref]

                    for a in getARefNames(info.rhs):
                        __remove(stmt_tab, ref_map, a)
                else:
                    ref_map[aref] -= 1

        mult_arrs = set([])
        for m in mults:
            mult_arrs |= set(getARefNames(m))

        def_map = {}
        for m_a in mult_arrs:
            if (self.stmt_tab.arr_map.has_key(m_a)):
                info = self.stmt_tab.arr_map[m_a]
                intm_infos = [info]
                for i_a in info.i_arr_name_set:
                    intm_infos.append(self.stmt_tab.arr_map[i_a])
                def_map[m_a] = intm_infos

        ref_map = __makeRefMap(self.stmt_tab, self.top_arefs)
        for a in i_arrs:
            __remove(self.stmt_tab, ref_map, a)
            self.top_arefs.remove(a)

        return def_map


def getARefNames(e):
    arefs = []
    if (isinstance(e, Array)):
        arefs.append(e.name)
    elif (isinstance(e, Addition) or isinstance(e, Multiplication)):
        for se in e.subexps:
            arefs.extend(getARefNames(se))
    else:
        raise OptimizerError('%s: unknown expression' % __name__)
    return arefs


def compareInterNames(n1, n2):
    num1 = int(n1[2:])
    num2 = int(n2[2:])
    if num1 > num2:
        return 1
    else:
        return -1

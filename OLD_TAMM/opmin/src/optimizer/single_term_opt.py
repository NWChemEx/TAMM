from combination import getAllCombinations, getTwoCombinations
from error import OptimizerError
from ast.absyn import Array, Multiplication
from ast.absyn_lib import renameIndices, buildRenamingTable
from stmt_table import StmtTable


def optimize(mult, stmt_tab, index_tab, volatile_tab, iteration):
    results = []
    coef = mult.coef

    exh_results = __exhaustiveSearch(
        mult.subexps, index_tab, volatile_tab, iteration, stmt_tab.shallow_copy(), mult.sym_groups)

    for (exh_i_arr, exh_info, exh_stmt_tab) in exh_results:
        exh_i_arr.multiplyCoef(coef)
        updated_stmt_tab = stmt_tab.shallow_copy()
        updated_stmt_tab.updateWithStmtTab(exh_stmt_tab, exh_info.i_arr_name_set | set([exh_i_arr.name]))
        results.append((exh_i_arr, exh_info, updated_stmt_tab))

    return results


def __exhaustiveSearch(subexps, index_tab, volatile_tab, iteration, stmt_tab, global_sym_groups):
    size = len(subexps)  # size = number of tensors, subexps = tensors

    if (size == 2):
        cur_subexps = [se.replicate() for se in subexps]
        mult = Multiplication(cur_subexps)
        mult.setOps(global_sym_groups)
        info = stmt_tab.getInfoForExp(mult, index_tab, volatile_tab, iteration)
        i_arr = info.lhs.replicate()
        ren_tab = buildRenamingTable(info.from_inds, info.to_inds)
        renameIndices(i_arr, ren_tab)
        return [(i_arr, info, stmt_tab)]

    old_i_arr_name_set = set(stmt_tab.getAllIntermediateNames())
    storage_tab = {}
    combinations = getAllCombinations(size)

    for cur_size in range(1, size+1):  # size = 3 --> cur_size = 1,2,3
        if (cur_size > 2):
            subcombinations = getAllCombinations(cur_size)

        for cur_comb in combinations[cur_size]:
            if (cur_size == 1):
                i_arr = subexps[cur_comb[0]].replicate()
                storage_tab[str(cur_comb)] = __ResultInfo([i_arr], None)
            elif (cur_size == 2):
                cur_subexps = map(lambda x: subexps[x].replicate(), cur_comb)
                mult = Multiplication(cur_subexps)
                mult.setOps(global_sym_groups)
                info = stmt_tab.getInfoForExp(mult, index_tab, volatile_tab, iteration)
                i_arr = info.lhs.replicate()
                ren_tab = buildRenamingTable(info.from_inds, info.to_inds)
                renameIndices(i_arr, ren_tab)
                storage_tab[str(cur_comb)] = __ResultInfo([i_arr], None)
            else:
                i_arrs = None
                infos = None
                cost = None

                for left_size in range(1, (cur_size/2)+1):   # 3 -> 1+2,2+1    4->1+3,2+2,3+1
                    right_size = cur_size - left_size

                    left_combs = subcombinations[left_size]
                    right_combs = subcombinations[right_size][:]
                    right_combs.reverse()

                    if (left_size == right_size):
                        up_bound = len(left_combs) / 2      # [0,1]+[2,3] = [2,3][0,1]
                    else:
                        up_bound = len(left_combs)

                    for i in range(0, up_bound):
                        left_key = map(lambda x: cur_comb[x], left_combs[i])    # find left key in cur_comb
                        right_key = map(lambda x: cur_comb[x], right_combs[i])  # find right key in cur_comb

                        left_res_info = storage_tab[str(left_key)]
                        right_res_info = storage_tab[str(right_key)]

                        for a1 in left_res_info.i_arrs:
                            for a2 in right_res_info.i_arrs:
                                mult = Multiplication([a1.replicate(), a2.replicate()])
                                mult.setOps(global_sym_groups)
                                info = stmt_tab.getInfoForExp(mult, index_tab, volatile_tab, iteration)
                                new_i_arr_name_set = (set([info.lhs.name]) | info.i_arr_name_set) - old_i_arr_name_set
                                cur_cost = stmt_tab.getCostForIntermediateNames(new_i_arr_name_set)

                                if (i_arrs is None or cur_cost <= cost):
                                    i_arr = info.lhs.replicate()
                                    ren_tab = buildRenamingTable(info.from_inds, info.to_inds)
                                    renameIndices(i_arr, ren_tab)
                                    if (i_arrs is None or cur_cost < cost):
                                        i_arrs = [i_arr]
                                        infos = [info]
                                        cost = cur_cost
                                    elif (cur_cost == cost):
                                        i_arrs.append(i_arr)
                                        infos.append(info)
                storage_tab[str(cur_comb)] = __ResultInfo(i_arrs, infos)

    res_info = storage_tab[str(combinations[size][0])]
    stmt_tabs = [stmt_tab] * len(res_info.i_arrs)
    return zip(res_info.i_arrs, res_info.infos, stmt_tabs)


class __ResultInfo:

    def __init__(self, i_arrs, infos):
        self.i_arrs = i_arrs
        self.infos = infos


def optimizeHeuristically(mult, stmt_tab, index_tab, volatile_tab, iteration):
    coef = mult.coef
    orig_stmt_tab = stmt_tab
    stmt_tab = stmt_tab.shallow_copy()
    old_i_arr_name_set = set(stmt_tab.getAllIntermediateNames())

    best_results = None
    best_cost = None

    greedy_trees = __greedySearch(
        mult.subexps, index_tab, volatile_tab, iteration, stmt_tab, mult.sym_groups, old_i_arr_name_set)

    for cur_tree in greedy_trees:
        (cur_i_arr, cur_info) = __flattenTree(cur_tree, index_tab, volatile_tab, iteration, stmt_tab, mult.sym_groups)
        new_i_arr_name_set = (set([cur_info.lhs.name]) | cur_info.i_arr_name_set) - old_i_arr_name_set
        cur_cost = stmt_tab.getCostForIntermediateNames(new_i_arr_name_set)

        if (best_results is None or cur_cost < best_cost):
            best_results = [(cur_i_arr, cur_info, cur_tree)]
            best_cost = cur_cost
        elif (cur_cost == best_cost):
            best_results.append((cur_i_arr, cur_info, cur_tree))

    iter_trees = greedy_trees

    while (True):
        one_move_results = None
        one_move_cost = None

        close_trees = __getNeighborsWithinDistance(iter_trees, 1, index_tab, volatile_tab, iteration, mult.sym_groups)
        for cur_tree in close_trees:
            (cur_i_arr, cur_info) = __flattenTree(
                cur_tree, index_tab, volatile_tab, iteration, stmt_tab, mult.sym_groups)
            new_i_arr_name_set = (set([cur_info.lhs.name]) | cur_info.i_arr_name_set) - old_i_arr_name_set
            cur_cost = stmt_tab.getCostForIntermediateNames(new_i_arr_name_set)

            if (one_move_results is None or cur_cost < one_move_cost):
                one_move_results = [(cur_i_arr, cur_info, cur_tree)]
                one_move_cost = cur_cost
            elif (cur_cost == one_move_cost):
                one_move_results.append((cur_i_arr, cur_info, cur_tree))

        if (one_move_results is None or one_move_cost > best_cost):
            break
        elif (one_move_cost == best_cost):
            best_results.extend(one_move_results)
            iter_trees = [tup[2] for tup in best_results]
        else:
            assert(one_move_cost < best_cost), '%s: one step cost must be less than the current best cost' % __name__
            best_results = one_move_results
            best_cost = one_move_cost
            iter_trees = [tup[2] for tup in one_move_results]

    results = []
    for (heu_i_arr, heu_info, heu_tree) in best_results:
        heu_i_arr.multiplyCoef(coef)
        updated_stmt_tab = orig_stmt_tab.shallow_copy()
        updated_stmt_tab.updateWithStmtTab(stmt_tab, heu_info.i_arr_name_set | set([heu_i_arr.name]))
        results.append((heu_i_arr, heu_info, updated_stmt_tab))
    return results


def __factorial(n):
    t = 1.0
    while n > 1:
        t *= n
        n -= 1
    return t


def __binomial(n, k):
    t = 1.0
    for i in range(k):
        t *= (n - i)
    t /= __factorial(k)
    return t


def __getNumOfElements(index_tab, sym_groups):
    t = 1.0
    for g in sym_groups:
        r = index_tab[g[0]]
        t *= __binomial(r, len(g))
    return t


def __greedySearch(trees, index_tab, volatile_tab, iteration, stmt_tab, global_sym_groups, i_arr_name_set):
    size = len(trees)

    if (size == 2):
        r_trees = [t.replicate() for t in trees]
        tree = Multiplication(r_trees)
        tree.setOps(global_sym_groups)
        return [tree]

    best_trees = None
    best_combs = None
    best_profit = None

    for cur_comb in getTwoCombinations(size):
        tree = Multiplication(map(lambda x: trees[x].replicate(), cur_comb))
        tree.setOps(global_sym_groups)
        (i_arr, info) = __flattenTree(tree, index_tab, volatile_tab, iteration, stmt_tab, global_sym_groups)

        loss = __getNumOfElements(index_tab, tree.sym_groups)
        profit = __getNumOfElements(index_tab, tree.sum_sym_groups)

        if (info.is_volatile):
            loss /= iteration
            profit /= iteration

        is_common = i_arr.name in i_arr_name_set
        if (is_common):
            cur_profit = profit + loss
        else:
            cur_profit = profit - loss

        if (best_trees is None or cur_profit > best_profit):
            best_trees = [tree]
            best_combs = [cur_comb]
            best_profit = cur_profit
        elif (cur_profit == best_profit):
            best_trees.append(tree)
            best_combs.append(cur_comb)

    results = []
    for (tree, comb) in zip(best_trees, best_combs):
        cur_trees = trees[:]
        del cur_trees[comb[1]]
        del cur_trees[comb[0]]
        cur_trees.append(tree)

        results.extend(
            __greedySearch(cur_trees, index_tab, volatile_tab, iteration, stmt_tab, global_sym_groups, i_arr_name_set)
        )
    return results


def __getNeighborsWithinDistance(trees, distance, index_tab, volatile_tab, iteration, global_sym_groups):
    if (distance < 0):
        raise OptimizerError('%s: neighbor distance cannot be negative' % __name__)

    if (distance == 0):
        return []

    stmt_tab = StmtTable()
    stmt_tab.turnOnCommonSubexp()

    results = []
    visited = {}

    iter_trees = trees
    for i in range(0, distance):
        next_iter_trees = []
        for cur_tree in iter_trees:
            (cur_i_arr, cur_info) = __flattenTree(
                cur_tree, index_tab, volatile_tab, iteration, stmt_tab, global_sym_groups)
            if (not visited.has_key(cur_i_arr.name)):
                visited[cur_i_arr.name] = None
                if (i > 0):
                    results.append(cur_tree)
                next_iter_trees.extend(__getAdjacentNeighbors(cur_tree, global_sym_groups))
        iter_trees = next_iter_trees

    for cur_tree in iter_trees:
        (cur_i_arr, cur_info) = __flattenTree(cur_tree, index_tab, volatile_tab, iteration, stmt_tab, global_sym_groups)
        if (not visited.has_key(cur_i_arr.name)):
            visited[cur_i_arr.name] = None
            results.append(cur_tree)

    return results


def __getAdjacentNeighbors(tree, global_sym_groups):
    if (not isinstance(tree, Multiplication)):
        return []

    adj_neighbors = []

    for n in __getAdjacentNeighbors(tree.subexps[0], global_sym_groups):
        m = Multiplication([n, tree.subexps[1].replicate()])
        m.setOps(global_sym_groups)
        adj_neighbors.append(m)

    for n in __getAdjacentNeighbors(tree.subexps[1], global_sym_groups):
        m = Multiplication([tree.subexps[0].replicate(), n])
        m.setOps(global_sym_groups)
        adj_neighbors.append(m)

    if (isinstance(tree.subexps[0], Multiplication)):
        left_se = Multiplication([tree.subexps[1].replicate(), tree.subexps[0].subexps[1].replicate()])
        left_se.setOps(global_sym_groups)
        right_se = tree.subexps[0].subexps[0].replicate()
        m = Multiplication([left_se, right_se])
        m.setOps(global_sym_groups)
        adj_neighbors.append(m)

        left_se = Multiplication([tree.subexps[0].subexps[0].replicate(), tree.subexps[1].replicate()])
        left_se.setOps(global_sym_groups)
        right_se = tree.subexps[0].subexps[1].replicate()
        m = Multiplication([left_se, right_se])
        m.setOps(global_sym_groups)
        adj_neighbors.append(m)

    if (isinstance(tree.subexps[1], Multiplication)):
        left_se = tree.subexps[1].subexps[0].replicate()
        right_se = Multiplication([tree.subexps[0].replicate(), tree.subexps[1].subexps[1].replicate()])
        right_se.setOps(global_sym_groups)
        m = Multiplication([left_se, right_se])
        m.setOps(global_sym_groups)
        adj_neighbors.append(m)

        left_se = tree.subexps[1].subexps[1].replicate()
        right_se = Multiplication([tree.subexps[1].subexps[0].replicate(), tree.subexps[0].replicate()])
        right_se.setOps(global_sym_groups)
        m = Multiplication([left_se, right_se])
        m.setOps(global_sym_groups)
        adj_neighbors.append(m)

    return adj_neighbors


def __flattenTree(tree, index_tab, volatile_tab, iteration, stmt_tab, global_sym_groups):
    both_array = True

    arr_subexps = []
    for se in tree.subexps:
        if (isinstance(se, Array)):
            arr_subexps.append(se)
        elif (isinstance(se, Multiplication)):
            both_array = False
            (i_arr, info) = __flattenTree(se, index_tab, volatile_tab, iteration, stmt_tab, global_sym_groups)
            arr_subexps.append(i_arr)
        else:
            raise OptimizerError('%s: unexpected expression type' % __name__)

    if (not both_array):
        tree = Multiplication(arr_subexps)
        tree.setOps(global_sym_groups)

    info = stmt_tab.getInfoForExp(tree, index_tab, volatile_tab, iteration)
    i_arr = info.lhs.replicate()
    ren_tab = buildRenamingTable(info.from_inds, info.to_inds)
    renameIndices(i_arr, ren_tab)

    return (i_arr, info)

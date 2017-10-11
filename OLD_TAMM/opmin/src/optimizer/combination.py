__storage_tab_1 = {}
__storage_tab_2 = {}


# Example:
#  Calling getAllCombinations(4) will return the following:
#
#  {1: [[0], [1], [2], [3]],
#   2: [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
#   3: [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
#   4: [[0, 1, 2, 3]]}

def getAllCombinations(size):
    if (__storage_tab_1.has_key(size)):
        return __storage_tab_1[size]
    else:
        all_combs = __generateAllCombinations(size)
        __storage_tab_1[size] = all_combs
        return all_combs


def __generateAllCombinations(size):
    sub_tab = {}
    for i in range(0, size):
        if (i == 0):
            sub_tab[1] = map(lambda x: [x], range(0, size))
        else:
            combs = []
            for c in sub_tab[i]:
                up_bound = c[len(c)-1]
                for e in range(up_bound+1, size):
                    comb = c[:]
                    comb.append(e)
                    combs.append(comb)
            sub_tab[i+1] = combs
    return sub_tab


# Example:
#  Calling getTwoCombinations(4) will return the following:
#
#  [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

def getTwoCombinations(size):
    if (__storage_tab_2.has_key(size)):
        return __storage_tab_2[size]
    else:
        two_combs = __generateTwoCombinations(size)
        __storage_tab_2[size] = two_combs
        return two_combs


def __generateTwoCombinations(size):
    two_combs = []
    one_combs = map(lambda x: [x], range(0, size))
    for c in one_combs:
        for e in range(c[0]+1, size):
            comb = c[:]
            comb.append(e)
            two_combs.append(comb)
    return two_combs

import random

O=3
V=3

def main():
    dims = [O,O,V,V]
    sym_group = [[0,1],[2,3]]
    #sym_group = []
    arr = makeArray(dims,True)
    arr_sym = replicateSymArray(arr,dims,sym_group)
    #print 'arr'
    #print arr
    #print
    #print arr_sym
    #for O1 in arr_sym:
    #    for O2 in O1:
    #        print O2
    #    print
def replicateSymArray(arrN, dims, sym_group):
    arr = nDim2oneDim(arrN)
    #print arr
    for x in range(0,len(arr)):
        nx = oneIdx2nIdx(x,dims)
        nx2 = sortnIdx(nx,sym_group)
        #sign = countParitySign(nx,nx2)
        y = nIdx2oneIdx(nx2,dims)
        #print x, nx, nx2, y
        sign = 1
        for g in sym_group:
            group = map(lambda z: nx[z], g)
            group2 = map(lambda z: nx2[z], g)
            sign *= countParitySign(group,group2)

        arr[x] = sign * arr[y]

        for g in sym_group:
            group = map(lambda z: nx2[z], g)
            if hasRepeat(group):
                arr[x]=0
    return oneDim2nDim(arr,dims)[0]

def makeSymArray(dims, sym_group, is_random=False, init_val = 0):
    arr = createArray(dims, is_random, init_val)
    #print arr
    #print len(arr)
    for x in range(0,len(arr)):
        nx = oneIdx2nIdx(x,dims)
        nx2 = sortnIdx(nx,sym_group)
        #sign = countParitySign(nx,nx2)
        y = nIdx2oneIdx(nx2,dims)
        #print x, nx, nx2, y
        sign = 1
        for g in sym_group:
            group = map(lambda z: nx[z], g)
            group2 = map(lambda z: nx2[z], g)
            sign *= countParitySign(group,group2)

        arr[x] = sign * arr[y]

        for g in sym_group:
            group = map(lambda z: nx2[z], g)
            if hasRepeat(group):
                arr[x]=0

    return oneDim2nDim(arr,dims)[0]

def hasRepeat(list):
    list2 = list[:]
    list2.sort()
    for x in range(1,len(list2)):
        if list2[x]==list2[x-1]: return True
    return False

def sortnIdx(nidx,sym_group):
    #print nidx, sym_group
    nidx2 = nidx[:]
    for g in sym_group:
        group = map(lambda x: nidx[x], g)
        group.sort(reverse=True) # descending
        for x in range(0,len(g)):
            nidx2[g[x]]=group[x]
    #print nidx2
    return nidx2

def countParitySign(ls1, ls2):
    ls2 = ls2[:] # copy ls2 for temporary use
    parity = 0
    for i in range(len(ls1)):
        if (ls1[i] != ls2[i]):
            #print ls1, ls2
            swap_i = ls2.index(ls1[i])
            ls2[swap_i] = ls2[i]
            ls2[i] = ls1[i]
            parity += 1
            #print '-->',ls1, ls2
    #print
    #print parity
    if (parity % 2 == 0):
        return 1
    else:
        return -1

def createArray(dims, is_random, init_val):
    array = []
    element_count = 1
    for x in range(0,len(dims)):
        element_count *= dims[x]
    for x in range(0,element_count):
        if is_random: array.append(random.randint(1,9))
        else: array.append(init_val)
    return array

def oneDim2nDim(array,dims):
    result = []
    count = dims[-1]
    temp=[]
    for x in range(0,len(array)):
        temp.append(array[x])
        if x%count==count-1:
            result.append(temp)
            temp=[]
    if len(dims)>1: return oneDim2nDim(result,dims[:-1])
    else: return result

def nDim2oneDim(array):
    #print 'array',array
    result = []
    if isinstance(array[0],list):
        for x in range(0,len(array)):
            result.extend(nDim2oneDim(array[x]))
    else:
        return array
    #print 'result',result
    return result

def oneIdx2nIdx(idx,dims): # idx = 5 , dims = [2,2,2] ==> nidx = [1,0,1]
    if len(dims)==0: return []
    else:
        x = idx%dims[-1]  # dims[-1]!=0
        nidx = oneIdx2nIdx(idx/dims[-1],dims[:-1])
        nidx.append(x)
    return nidx

def nIdx2oneIdx(nidx,dims): # nidx = [1,0,0] , dims = [2,2,2] ==> idx = 8
    sum = 0
    for x in range(0,len(dims)):
        factor = 1
        for y in range(x+1,len(dims)):
            factor *= dims[y]
        sum += nidx[x]*factor
    return sum

def makeArray(dimensions, is_random = False, init_val = 0):
    arr = []
    bucket = [arr]
    for (i, d) in enumerate(dimensions):
        next_bucket = []
        for e in bucket:
            for n in range(0, d):
                if (i == len(dimensions) - 1):
                    if (is_random):
                        rnum = random.randint(1,5)
                        e.append(rnum)
                    else:
                        e.append(init_val)
                else:
                    new_list = []
                    e.append(new_list)
                    next_bucket.append(new_list)
        bucket = next_bucket
    if (len(arr) == 0):
        if (is_random):
            rnum = random.randint(1,5)
            arr = rnum
        else:
            arr = init_val
    return arr

def replicateArray(arr):
    if (not isinstance(arr, list)):
        return arr
    r_arr = []
    for i in arr:
        r_arr.append(replicateArray(i))
    return r_arr

#main()

import single_term_opt
from time import time
from error import OptimizerError
from ast.absyn import ArrayDecl, VolatileDecl, IterationDecl, RangeDecl, IndexDecl, \
     AssignStmt, Array, Addition, Multiplication
from cost_model import countOpCost
from ast.absyn_lib import renameIndices, buildRenamingTable, extendRenamingTable
from stmt_table import StmtTable, UsageTable
from combination import getTwoCombinations
from refine import updateCSE, updateFAC, refineTop
from math import gcd
from backend.unparser import unparseCompElem

#--------------------------------------------------------------------

__USE_CSE = False
__USE_FACT = False
__USE_REFINE = False

#--------------------------------------------------------------------

__CSE_INTERVAL = 10
__CSE_SOFT_LIMIT = 5
__CSE_HARD_LIMIT = 10

#--------------------------------------------------------------------

__FACT_CHUNK_SIZE = 10
__FACT_MAX_CHUNK_SIZE = 50
__FACT_LIMIT = 5

#--------------------------------------------------------------------

def optimize(trans_unit, use_cse, use_fact, use_refine):

    # set optimization options
    global __USE_CSE, __USE_FACT, __USE_REFINE
    __USE_FACT = use_fact
    __USE_REFINE = use_refine
    __USE_CSE = use_cse
    if use_refine: __USE_CSE = False

    # optimize each compound elements
    for ce in trans_unit.comp_elems:        # ce = { a set of dec and stmt }
        __optimizeCompElem(ce)

def __optimizeCompElem(comp_elem):

    # collect the index range values, range indices, volatile arrays, and iteration count
    index_tab = __collectIndexInfo(comp_elem)       # {'h1':'10.0','h2':'500.0',etc.}
    range_tab = __collectRangeInfo(comp_elem)       # {'h1':'O','h2':'V',etc.}
    volatile_tab = __collectVolatileInfo(comp_elem)
    iteration = __collectIterationInfo(comp_elem)
    
    #print index_tab
    #print range_tab
    #print volatile_tab
    #print iteration
    
    # print cost of original equation
    org_cost = __applyOperationCount(comp_elem, range_tab, index_tab, volatile_tab, iteration)
    print '----------------------------------------'
    print '--> total cost of original equation = %s' % org_cost
    print '----------------------------------------'

    # apply optimizations
    cost = __applyOptimizations(comp_elem, range_tab, index_tab, volatile_tab, iteration)

    print '----------------------------------------'
    print '--> total cost of optimized equation = %s' % cost
    print '----------------------------------------'

#--------------------------------------------------------------------

def __collectIndexInfo(comp_elem):
    range_tab = {}
    index_tab = {}
    for e in comp_elem.elems:
        if (isinstance(e, RangeDecl)):
            range_tab[e.name] = e.value
        elif (isinstance(e, IndexDecl)):
            index_tab[e.name] = range_tab[e.range]
    return index_tab

def __collectRangeInfo(comp_elem):
    range_tab = {}
    for e in comp_elem.elems:
        if (isinstance(e, IndexDecl)):
            range_tab[e.name] = e.range
    return range_tab

def __collectVolatileInfo(comp_elem):
    volatile_tab = {}
    for e in comp_elem.elems:
        if (isinstance(e, VolatileDecl)):
            volatile_tab[e.arr] = None
    return volatile_tab

def __collectIterationInfo(comp_elem):
    iteration = 1
    for e in comp_elem.elems:
        if (isinstance(e, IterationDecl)):
            iteration = e.value 
    return iteration

#--------------------------------------------------------------------
def __applyOperationCount(comp_elem, range_tab, index_tab, volatile_tab, iteration):
    stmt_tab = StmtTable()
    total_cost = __countTotalCost(comp_elem, stmt_tab, index_tab, volatile_tab, iteration)
    return total_cost


def __applyOptimizations(comp_elem, range_tab, index_tab, volatile_tab, iteration):

    # wrap terms
    __wrapTerms(comp_elem)
    
    # collect terms
    terms = __collectTerms(comp_elem)
    
    print '----------------------------------------'
    print '--> total number of terms: %s' % len(terms)
    print '-------- starts weight-counting --------'
    t1 = time()

    # create statement table
    stmt_tab = StmtTable()
    stmt_tab.turnOffCommonSubexp()

    # apply single-term optimization to count weight for each term
    for (i, t) in enumerate(terms):
        print '... count weight for %s-th term (out of %s)' % (i+1, len(terms))
        
        cur_stmt_tab = StmtTable()
        cur_stmt_tab.turnOffCommonSubexp()
        cur_results = single_term_opt.optimize(t.mult, cur_stmt_tab, index_tab, volatile_tab, iteration)
        (cur_i_arr, cur_info, cur_stmt_tab) = cur_results[0]
        cur_i_arr_names = cur_stmt_tab.getAllIntermediateNames()

        #print cur_i_arr_names
        #print cur_stmt_tab

        t.weight = cur_stmt_tab.getCostForIntermediateNames(cur_i_arr_names)
        t.optimized_i_arr = cur_i_arr
        t.optimized_info = cur_info

        stmt_tab.updateWithStmtTab(cur_stmt_tab, cur_i_arr_names)

    t2 = time()
    print '--> weight-counting time: %s secs' % (t2-t1)
    print '-------- finished weight-counting --------'
    
    # sort terms from heaviest to lightest
    #terms.sort(lambda x,y: -cmp(x.weight, y.weight))

    if (__USE_CSE):

        # re-create statement table
        stmt_tab = StmtTable()
        stmt_tab.turnOnCommonSubexp()

        print '-------- starts common-subexpression elimination --------'
        t1 = time()

        # common-subexpression elimination
        results = __cse(terms, stmt_tab, index_tab, volatile_tab, iteration,
                        __CSE_INTERVAL, __CSE_SOFT_LIMIT, __CSE_HARD_LIMIT, True)

        t2 = time()
        print '--> time for common-subexpression identification: %s secs' % (t2-t1)
        print '-------- finished common-subexpression elimination --------'

        # update terms with CSE results
        #print results[0]
        (cur_i_arrs, cur_infos, cur_stmt_tab) = results[0] 
        stmt_tab = cur_stmt_tab
        for (cur_term, cur_i_arr, cur_info) in zip(terms, cur_i_arrs, cur_infos):
            cur_term.optimized_i_arr = cur_i_arr
            cur_term.optimized_info = cur_info

    if (__USE_FACT):

        # collect additions and top-level array references
        adds = __collectAdditions(comp_elem)
        top_arefs = __collectTopARefNames(comp_elem)
        #print top_arefs

        # create usage table
        usage_tab = UsageTable(stmt_tab, top_arefs)

        print '-------- starts factorization --------'
        t1 = time()

        # factorization
        result = __factorize(adds, usage_tab, index_tab, volatile_tab, iteration)
        
        t2 = time()
        print '--> time for factorization: %s secs' % (t2-t1)
        print '-------- finished factorization --------'

        # update additions with factorization results
        (fact_multi_terms, fact_usage_tab) = result
        #print 'test',fact_multi_terms,fact_usage_tab.stmt_tab
        stmt_tab = fact_usage_tab.stmt_tab
        for (ad, mt) in zip(adds, fact_multi_terms):
            del ad.subexps[:]
            ad.subexps.extend(mt)
        for e in comp_elem.elems:
            if (isinstance(e, AssignStmt) and isinstance(e.rhs, Addition)):
                if (len(e.rhs.subexps) == 1):
                    e.rhs = e.rhs.subexps[0]
        
    # unwrap terms
    __unwrapTerms(comp_elem, stmt_tab, range_tab)
    
    # check for errors (optional)
    __sanityCheck(comp_elem, stmt_tab)
    
    # count total cost
    total_cost = __countTotalCost(comp_elem, stmt_tab, index_tab, volatile_tab, iteration)
    #print comp_elem
    #print total_cost
    
    if (__USE_REFINE):
        
        print 'cost before refinement', total_cost
        curr_cost = total_cost
        
        x=0
        while True:
            x=x+1
            comp_elem.reArrange()
            
            print 'start update CSE ...'
            updateCSE(comp_elem, stmt_tab)
            total_cost = __countCostForCompElem(comp_elem, stmt_tab, index_tab, volatile_tab, iteration, False)    
            print total_cost
            print 'end update CSE...'
            print
        
            #break

            print 'start update FAC ...'
            #new_comp_elem = comp_elem.replicate()
            #new_stmt_tab = stmt_tab.shallow_copy()
            #updateFAC(new_comp_elem, new_stmt_tab, index_tab, volatile_tab, iteration, range_tab)        
            #new_total_cost = __countCostForCompElem(new_comp_elem, new_stmt_tab, index_tab, volatile_tab, iteration)
            
            updateFAC(comp_elem, stmt_tab, index_tab, volatile_tab, iteration, range_tab)        
            new_total_cost = __countCostForCompElem(comp_elem, stmt_tab, index_tab, volatile_tab, iteration, False)
            comp_elem.reArrange()
            #print 'new_cost (used if better)', new_total_cost
            print 'end update FAC...'
            print
            if new_total_cost == total_cost: break
            
            #if new_total_cost != total_cost:
            #    comp_elem = new_comp_elem.replicate()
            #    stmt_tab = new_stmt_tab.shallow_copy()
            #    total_cost = new_total_cost
            #else: break
    
    
        comp_elem.removeRedundant()
        total_cost = __countCostForCompElem(comp_elem, stmt_tab, index_tab, volatile_tab, iteration, False)
        comp_elem.reArrange()
    	#while (comp_elem.reArrange()): pass
    	print
    	print 'start refine top'
    	while (True):
           success = refineTop(comp_elem,stmt_tab,index_tab, volatile_tab, iteration, range_tab)
           if not success: break
    	   print 'end refine top'
    	   print

    	total_cost = __countCostForCompElem(comp_elem, stmt_tab, index_tab, volatile_tab, iteration,False)
    
    	updateCSE(comp_elem, stmt_tab)
    	comp_elem.removeRedundant()
        comp_elem.reArrange()
        total_cost = __countCostForCompElem(comp_elem, stmt_tab, index_tab, volatile_tab, iteration,False)
    
    return total_cost

#--------------------------------------------------------------------



class Term:
    def __init__(self, mult):
        self.mult = mult
        self.weight = None
        self.optimized_i_arr = None
        self.optimized_info = None
        self.arr_symbols = None

    def replicate(self):
        t = Term(self.mult.replicate())
        t.mult = self.mult
        t.weight = self.weight
        t.optimized_i_arr = self.optimized_i_arr
        t.optimized_info = self.optimized_info
        t.arr_symbols = self.arr_symbols

        t.mult = t.mult.replicate()
        if (t.optimized_i_arr != None):
            t.optimized_i_arr = t.optimized_i_arr.replicate()
        if (t.arr_symbols != None):
            t.arr_symbols = t.arr_symbols[:]
        return t

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = '{'
        s += 'mult=' + str(self.mult) + ', \n'
        s += 'i_arr=' + str(self.optimized_i_arr)
        s += '}'
        return s
    
#--------------------------------------------------------------------

def __wrapTerms(comp_elem):
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt)):
            if (isinstance(e.rhs, Addition)):
                e.rhs = __wrapTermsAddition(e.rhs)
            elif (isinstance(e.rhs, Multiplication)):
                e.rhs = __wrapTermsMultiplication(e.rhs)

def __wrapTermsAddition(e):
    subexps = []
    for se in e.subexps:
        if (isinstance(se, Multiplication)):
            subexps.append(Term(se))
        else:
            subexps.append(se)
    e.subexps = subexps
    return e

def __wrapTermsMultiplication(e):
    return Term(e)
    
#--------------------------------------------------------------------

def __collectTerms(comp_elem):
    terms = []
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt)):
            if (isinstance(e.rhs, Addition)):
                for se in e.rhs.subexps:
                    if (isinstance(se, Term)):
                        terms.append(se)
            elif (isinstance(e.rhs, Term)):
                terms.append(e.rhs)
    return terms

def __collectAdditions(comp_elem):
    adds= []
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt) and isinstance(e.rhs, Addition)):
            adds.append(e.rhs)
    return adds

def __collectTopARefNames(comp_elem):
    top_arefs = []
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt)):
            if (isinstance(e.rhs, Array)):
                top_arefs.append(e.rhs.name)
            elif (isinstance(e.rhs, Addition)):
                for se in e.rhs.subexps:
                    if (isinstance(se, Array)):
                        top_arefs.append(se.name)
                    else:
                        assert(isinstance(se, Term)), '%s: must be a term' % __name__
                        top_arefs.append(se.optimized_i_arr.name)
            else:
                assert(isinstance(e.rhs, Term)), '%s: must be a term' % __name__
                top_arefs.append(e.rhs.optimized_i_arr.name)
    return top_arefs
                                
#--------------------------------------------------------------------

def __unwrapTerms(comp_elem, stmt_tab, range_tab):
    leading_arr_names = []
    n_elems = []
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt)):
            if (isinstance(e.rhs, Addition)):
                (add, leading_new_elems) = __unwrapTermsAddition(e.rhs, leading_arr_names, stmt_tab, range_tab)
                e.rhs = add
                n_elems.extend(leading_new_elems)
            elif (isinstance(e.rhs, Term)):
                (mult, leading_new_elems) = __unwrapTermsMultiplication(e.rhs, leading_arr_names, stmt_tab, range_tab)
                e.rhs = mult
                n_elems.extend(leading_new_elems)
        n_elems.append(e)
    comp_elem.elems = n_elems

def __unwrapTermsAddition(e, leading_arr_names, stmt_tab, range_tab):
    leading_elems = []
    subexps = []

    trailing_arr_names = set([])    
    for se in e.subexps:
        if (isinstance(se, Term)):
            trailing_arr_names |= (se.optimized_info.i_arr_name_set | set([se.optimized_info.lhs.name]))
            subexps.append(se.optimized_i_arr.replicate())
        else:
            subexps.append(se)

    #print subexps
    #print trailing_arr_names
    #print leading_arr_names

    trailing_arr_names -= set(leading_arr_names)
    ordered_infos = stmt_tab.getOrderedInfos(leading_arr_names, trailing_arr_names)

    asgs = []
    for i in ordered_infos:
        leading_elems.append(__createArrayDecl(i.lhs, range_tab))
        asgs.append(AssignStmt(i.lhs.replicate(), i.rhs.replicate()))
        leading_arr_names.append(i.lhs.name)
    leading_elems.extend(asgs)
    
    e.subexps = subexps
    
    return (e, leading_elems)
    

def __unwrapTermsMultiplication(e, leading_arr_names, stmt_tab, range_tab):
    leading_elems = []

    trailing_arr_names = (e.optimized_info.i_arr_name_set | set([e.optimized_info.lhs.name])) - set(leading_arr_names)
    ordered_infos = stmt_tab.getOrderedInfos(leading_arr_names, trailing_arr_names)

    asgs = []
    for i in ordered_infos:
        leading_elems.append(__createArrayDecl(i.lhs, range_tab))
        asgs.append(AssignStmt(i.lhs.replicate(), i.rhs.replicate()))
        leading_arr_names.append(i.lhs.name)
    leading_elems.extend(asgs)
        
    #print 'e',e
    #print 'le',leading_elems
    
    return (e.optimized_i_arr.replicate(), leading_elems)

#---------------------------------------------------------------------

def __createArrayDecl(arr, range_tab):
    upper_ranges = map(lambda x: range_tab[x], arr.upper_inds)
    lower_ranges = map(lambda x: range_tab[x], arr.lower_inds)
    inds = arr.upper_inds + arr.lower_inds
    #print inds, arr.name
    #print arr.sym_groups
    #print arr.vsym_groups
    #print
    sym_groups = [map(lambda x: inds.index(x), g) for g in arr.sym_groups]
    vsym_groups = [map(lambda x: inds.index(x), g) for g in arr.vsym_groups]
    for g in sym_groups:
        g.sort()
    sym_groups.sort()
    return ArrayDecl(arr.name, upper_ranges, lower_ranges, sym_groups, vsym_groups)

#--------------------------------------------------------------------

def __sanityCheck(comp_elem, stmt_tab):
    all_i_arr_names = stmt_tab.getAllIntermediateNames()
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt)):
            if (e.lhs.name in all_i_arr_names):
                all_i_arr_names.remove(e.lhs.name)
    if (len(all_i_arr_names) > 0):
        raise OptimizerError('%s: some intermediates in statement table are not produced' % __name__)

    previous_lhs_names = []
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt)):
            if (isinstance(e.rhs, Array)):
                if (e.rhs.name.startswith(StmtTable.IARRAY_PREFIX) and (e.rhs.name not in previous_lhs_names)):
                    raise OptimizerError('%s: no definition of array "%s" before its reference' % (__name__, e.rhs.name))
            else:
                assert(isinstance(e.rhs, Addition) or isinstance(e.rhs, Multiplication)), '%s: unknown expression' % __name__
                for se in e.rhs.subexps:
                    assert(isinstance(se, Array)), '%s: operand must be an array' % __name__
                    if (se.name.startswith(StmtTable.IARRAY_PREFIX) and (se.name not in previous_lhs_names)):
                        raise OptimizerError('%s: no definition of array "%s" before its reference' % (__name__, se.name))
            previous_lhs_names.append(e.lhs.name)

#--------------------------------------------------------------------

def __countTotalCost(comp_elem, stmt_tab, index_tab, volatile_tab, iteration):
    volatile_list = stmt_tab.getVolatileIntermediateNames(stmt_tab.getAllIntermediateNames())
    updated_volatile_tab = volatile_tab.copy()
    for v in volatile_list:
        updated_volatile_tab[v] = None

    all_i_arr_names = stmt_tab.getAllIntermediateNames()

    cost = 0.0
    #print cost
    
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt) and (e.lhs.name not in all_i_arr_names)):
            (cur_op_cost, cur_is_volatile) = countOpCost(e.rhs, index_tab, updated_volatile_tab, iteration)
            cost += cur_op_cost
            #print cur_op_cost,e
            
    cost += stmt_tab.getCostForIntermediateNames(all_i_arr_names)
    return cost    

def __countCostForCompElem(comp_elem, stmt_tab, index_tab, volatile_tab, iteration, show):

    volatile_list = stmt_tab.getVolatileIntermediateNames(stmt_tab.getAllIntermediateNames())
    updated_volatile_tab = volatile_tab.copy()
    for v in volatile_list:
        updated_volatile_tab[v] = None

    cost = 0.0
    for e in comp_elem.elems:
        if (isinstance(e, AssignStmt)):
            (cur_op_cost, cur_is_volatile) = countOpCost(e.rhs, index_tab, updated_volatile_tab, iteration)
            #if show: print e.lhs.name, cur_op_cost
            cost += cur_op_cost
    #if show: print cost
    return cost
    
#---------------------------------------------------------------------
# Common Subexpression Elimination
#---------------------------------------------------------------------

def __cse(terms, stmt_tab, index_tab, volatile_tab, iteration, interval, soft_limit, hard_limit, show_message):
    acc = [([], [], stmt_tab, 0)]
    results = __cseAPS(terms, index_tab, volatile_tab, iteration, interval, soft_limit, hard_limit, show_message, acc)

    best_results = None
    best_cost = None
    for (cur_i_arrs, cur_infos, cur_stmt_tab, cur_cost) in results:
        if (best_results == None or cur_cost < best_cost):
            best_results = [(cur_i_arrs, cur_infos, cur_stmt_tab)]
            best_cost = cur_cost
        elif (cur_cost == best_cost):
            best_results.append((cur_i_arrs, cur_infos, cur_stmt_tab))
    return best_results
    
def __cseAPS(terms, index_tab, volatile_tab, iteration, interval, soft_limit, hard_limit, show_message, acc):
    if (len(terms) == 0):
        return acc

    if (show_message):
        print '... optimize %s-th term (out of %s)' % (len(acc[0][0])+1, len(acc[0][0])+len(terms))

    (cur_i_arrs, cur_infos, cur_stmt_tab, cur_cost) = acc[0]

    if (len(acc) > hard_limit):
        acc.sort(lambda x,y: cmp(x[3], y[3]))
        l1 = len(acc)
        acc = acc[:hard_limit]
        l2 = len(acc)
        if (show_message):
            print '--> exceed hard limit, cut search space (from %s to %s)' % (l1, l2)

    if (len(cur_i_arrs) > 0 and (len(cur_i_arrs) % interval == 0)):
        acc.sort(lambda x,y: cmp(x[3], y[3]))
        l1 = len(acc)
        acc = acc[:soft_limit]
        l2 = len(acc)
        if (show_message):
            print '--> periodic search-space reduction, cut search space (from %s to %s)' % (l1, l2)

    term = terms[0]
    next_acc = []

    for (cur_i_arrs, cur_infos, cur_stmt_tab, cur_cost) in acc:
        cur_results = single_term_opt.optimize(term.mult, cur_stmt_tab, index_tab, volatile_tab, iteration)
        for (cur_i_arr, cur_info, cur_updated_stmt_tab) in cur_results:
            next_i_arrs = cur_i_arrs + [cur_i_arr]
            next_infos = cur_infos + [cur_info]
            next_cost = cur_updated_stmt_tab.getCostForIntermediateNames(cur_updated_stmt_tab.getAllIntermediateNames())
            next_acc.append((next_i_arrs, next_infos, cur_updated_stmt_tab, next_cost))

    return __cseAPS(terms[1:], index_tab, volatile_tab, iteration, interval, soft_limit, hard_limit, show_message, next_acc)

#---------------------------------------------------------------------
# Factorization
#---------------------------------------------------------------------

def __factorize(adds, usage_tab, index_tab, volatile_tab, iteration):

    # copy multi terms
    multi_terms = [[se.replicate() for se in add.subexps] for add in adds]

    # add identical arrays
    #xxx
    
    # compute array symbols at each term
    for mt in multi_terms:
        for t in mt:
            if (isinstance(t, Term)):
                (mult, arr_symbols) = __symbolizeMult(t.mult)
                t.mult = mult
                t.arr_symbols = arr_symbols

    # factorization iteration
    fact_multi_terms = []
    fact_usage_tab = usage_tab
    for mt in multi_terms:
        cur_results = __divideAndConquerFactorization(mt, fact_usage_tab, index_tab, volatile_tab, iteration)
        (fact_multi_term, fact_usage_tab) = cur_results
        fact_multi_terms.append(fact_multi_term)

    # return factorized multi-terms
    return (fact_multi_terms, fact_usage_tab)
    
#---------------------------------------------------------------------

def __divideAndConquerFactorization(multi_term, usage_tab, index_tab, volatile_tab, iteration):

    # collect terms
    non_terms = filter(lambda x: not isinstance(x, Term), multi_term)
    terms = filter(lambda x: isinstance(x, Term), multi_term)

    # sort terms from heaviest to lightest
    terms.sort(lambda x,y: -cmp(x.weight, y.weight))

    # divide
    mterm_chunks = []
    i = 0
    j = __FACT_CHUNK_SIZE
    cur_chunk = terms[i:j]
    while (len(cur_chunk) > 0):
        mterm_chunks.append(cur_chunk)
        i += __FACT_CHUNK_SIZE
        j += __FACT_CHUNK_SIZE
        cur_chunk = terms[i:j]

    # factorization results
    fact_mterm_chunks = []
    fact_usage_tab = usage_tab

    while (True):

        # conquer
        n_mterm_chunks = []
        for (i, mt) in enumerate(mterm_chunks):

            print '*** factorize %s-th multi-term chunk (out of %s)' % (i+1, len(mterm_chunks))            
            
            cur_results = __factorizeMultiTerm(mt, fact_usage_tab, index_tab, volatile_tab, iteration, True,0)
            (fact_multi_term, fact_usage_tab) = cur_results[0]
            n_mterm_chunks.append(fact_multi_term)
        mterm_chunks = n_mterm_chunks

        if (len(mterm_chunks) <= 1):
            fact_mterm_chunks.extend(mterm_chunks)
            break

        # combine
        n_mterm_chunks = []
        for i in range(0, len(mterm_chunks), 2):
            j = i + 1
            if (j < len(mterm_chunks)):
                combined = mterm_chunks[i] + mterm_chunks[j]
                if (len(combined) <= __FACT_MAX_CHUNK_SIZE):
                    n_mterm_chunks.append(combined)
                else:
                    fact_mterm_chunks.append(combined)
            else:
                n_mterm_chunks.append(mterm_chunks[i])
        mterm_chunks = n_mterm_chunks
    
    # combine all factorized chunks
    terms = reduce(lambda x,y: x+y, fact_mterm_chunks, [])
    fact_multi_term = non_terms + terms
    return (fact_multi_term, fact_usage_tab)
    
#---------------------------------------------------------------------

def __factorizeMultiTerm(multi_term, usage_tab, index_tab, volatile_tab, iteration, show_message, level):

    #for zz in range(0, level):
    #    print'\t',
    #print multi_term
    
    # best results
    best_results = None
    best_cost = None

    # incrementally perform a two-term factorization at each iteration
    i = 0
    iter_results = [(multi_term, usage_tab)]
    while (len(iter_results) > 0):
        i += 1

        if (show_message):
            print '... %s-th two-term factorization performed' % (i)

        n_iter_results = []
        for (cur_multi_term, cur_usage_tab) in iter_results:
            one_step_results = __oneStepFactorizeMultiTerm(cur_multi_term, cur_usage_tab, index_tab, volatile_tab, iteration,level+1)
            for (one_step_multi_term, one_step_usage_tab, is_final) in one_step_results:
                #print
                #print 'osmt',one_step_multi_term
                #print 'osut',one_step_usage_tab.stmt_tab, is_final
                #print
                if (is_final):
                    cur_cost = __countCostForMultiTerm(one_step_multi_term, one_step_usage_tab,
                                                       index_tab, volatile_tab, iteration)
                                        
                    if (best_cost == None or cur_cost < best_cost):
                        best_results = [(one_step_multi_term, one_step_usage_tab)]
                        best_cost = cur_cost
                    elif (cur_cost == best_cost):
                        best_results.append((one_step_multi_term, one_step_usage_tab))
                else:
                    n_iter_results.append((one_step_multi_term, one_step_usage_tab))
                    
        iter_results = n_iter_results

        # cut search space
        if (len(iter_results) > __FACT_LIMIT):
            if (__FACT_LIMIT > 1):
                result_costs = map(lambda x: __countCostForMultiTerm(x[0], x[1], index_tab, volatile_tab, iteration),
                                   iter_results)
                compound = zip(iter_results, result_costs)
                compound.sort(lambda x,y: cmp(x[1], y[1]))
                iter_results = map(lambda x: x[0], compound)

            l1 = len(iter_results)
            iter_results = iter_results[:__FACT_LIMIT]
            l2 = len(iter_results)
            print '--> cut search space (from %s to %s)' % (l1, l2)
        
    # return best results
    return best_results

#---------------------------------------------------------------------

def __oneStepFactorizeMultiTerm(multi_term, usage_tab, index_tab, volatile_tab, iteration,level):

    # filter out the non-terms
    terms = []
    non_terms = []
    for t in multi_term:
        if (isinstance(t, Term)):
            terms.append(t.replicate())
        else:
            non_terms.append(t.replicate())

    #print non_terms
    #print terms
    #print

    # check if no factorization is possible
    if (len(terms) <= 1):
        n_multi_term = non_terms + terms
        n_usage_tab = usage_tab.shallow_copy()
        return [(n_multi_term, n_usage_tab, True)]

    # best two-term factorizations
    best_two_facts = None
    best_profit = None
    
    # find the best two-term factorization
    for cur_comb in getTwoCombinations(len(terms)):
        #print 'cur_comb',cur_comb
        (term1, term2) = map(lambda x: terms[x], cur_comb)
        (cur_results, cur_profit) = factorizeTwoTerms(term1, term2, usage_tab, index_tab, volatile_tab, iteration,level+1)

        if (cur_results != None):
            if (best_profit == None or cur_profit > best_profit):
                cur_results = [list(r) + [cur_comb] for r in cur_results]
                best_two_facts = cur_results
                best_profit = cur_profit
            elif (cur_profit == best_profit):  
                cur_results = [list(r) + [cur_comb] for r in cur_results]
                best_two_facts.extend(cur_results)

    # check if no factorization is possible
    if (best_two_facts == None):
        n_multi_term = non_terms + terms
        n_usage_tab = usage_tab.shallow_copy()
        return [(n_multi_term, n_usage_tab, True)]

    # compute array symbols at each factorized term
    for (cur_fact_term, cur_fact_usage_tab, cur_fact_comb) in best_two_facts:
        (mult, arr_symbols) = __symbolizeMult(cur_fact_term.mult)
        cur_fact_term.mult = mult
        cur_fact_term.arr_symbols = arr_symbols
        
    # compute one-step factorization results
    results = []
    for (fact_term, fact_usage_tab, fact_comb) in best_two_facts:
        n_non_terms = [t.replicate() for t in non_terms]
        n_terms = [t.replicate() for t in terms]
        del n_terms[fact_comb[1]]
        del n_terms[fact_comb[0]]
        n_terms.append(fact_term)
        n_multi_term = n_non_terms + n_terms
        n_usage_tab = fact_usage_tab
        
        results.append((n_multi_term, n_usage_tab, False))
    return results

#---------------------------------------------------------------------

def factorizeTwoTerms(term1, term2, usage_tab, index_tab, volatile_tab, iteration, level):
    
    #for zz in range(0, level):
    #    print'\t',
    #print 'start two term'
    # copy usage table
    usage_tab = usage_tab.shallow_copy()
    
    # count unfactorized cost
    volatile_list = usage_tab.stmt_tab.getVolatileIntermediateNames(usage_tab.stmt_tab.getAllIntermediateNames())
    updated_volatile_tab = volatile_tab.copy()
    for v in volatile_list:
        updated_volatile_tab[v] = None
    #print 't1 opt', term1.optimized_i_arr
    unfact_add = Addition([term1.optimized_i_arr, term2.optimized_i_arr])
    unfact_cost = usage_tab.stmt_tab.getCostForIntermediateNames(usage_tab.stmt_tab.getAllIntermediateNames())
    unfact_cost += countOpCost(unfact_add, index_tab, updated_volatile_tab, iteration)[0]

    # remove unnecessary intermediates out of usage table
    def_map = usage_tab.removeIntermediates([term1.optimized_i_arr.name, term2.optimized_i_arr.name], [term1.mult, term2.mult])
    #print 'def_map',def_map

    # best results
    best_results = None
    best_profit = None
    
    # search common array
    len1 = len(term1.arr_symbols)
    len2 = len(term2.arr_symbols)
    '''for i1 in range(0, len1):
        for i2 in range(0, len2):
            print 't1',term1.arr_symbols[i1]
            print 't2',term2.arr_symbols[i2]'''
    
    for i1 in range(0, len1):
        for i2 in range(0, len2):
            
            #print 'i1i2',i1,i2
            #print 'term1',term1
            #print 'term1.sym-->',term1.arr_symbols[i1]
            #print
            #print 'term2',term2
            #print 'term2.sym-->',term2.arr_symbols[i2]
            #print
            
            if (term1.arr_symbols[i1] == term2.arr_symbols[i2]):
                
                #print
                #print 'term1-->',term1
                #print 'term2-->',term2
                #print 'factor-->',term1.arr_symbols[i1]

                [g,c1,c2] = gcd(term1.mult.coef,term2.mult.coef)
                #if g==0:
                    #print term1.mult.coef, term1
                    #print term2.mult.coef, term2
                    #print
                #if term1.mult.coef==0: print term1
                #if term2.mult.coef==0: print term2
                
                    #print gcd,term1.mult.coef,term2.mult.coef
                #print gcd,c1,c2

                # break the 1st term
                right1 = [se.replicate() for se in term1.mult.subexps]
                left1 = right1.pop(i1)
                
                if (len(right1) == 1):
                    right1 = right1[0]
                else:
                    right1 = Multiplication(right1)
                    right1.setOps(term1.mult.sym_groups)
                #right1.multiplyCoef(term1.mult.coef)
                left1.multiplyCoef(g)
                #if (c1!=1.0):
                right1.multiplyCoef(c1)

                # break the 2nd term
                right2 = [se.replicate() for se in term2.mult.subexps]
                left2 = right2.pop(i2)
                
                if (len(right2) == 1):
                    right2 = right2[0]
                else:
                    right2 = Multiplication(right2)
                    right2.setOps(term2.mult.sym_groups)
                #right2.multiplyCoef(term2.mult.coef)
                left2.multiplyCoef(g)
                #if (c2!=1.0):
                right2.multiplyCoef(c2)

                # rename the 2nd term's components
                from_inds = left2.upper_inds + left2.lower_inds
                to_inds = left1.upper_inds + left1.lower_inds
                ren_tab = buildRenamingTable(from_inds, to_inds)
                mult_inds = term2.mult.upper_inds + term2.mult.lower_inds + term2.mult.sum_inds
                ren_tab = extendRenamingTable(ren_tab, set(mult_inds) - set(from_inds))
                #print ren_tab
                renameIndices(left2, ren_tab)
                renameIndices(right2, ren_tab)

                # remember important variables
                sym_groups = term1.mult.sym_groups
                factor = left1
                multi_term = [right1, right2]
                cur_usage_tab = usage_tab

                # prepare the multi term
                (multi_term, cur_usage_tab) = __prepareForRecursion(multi_term, def_map, cur_usage_tab,
                                                                    index_tab, volatile_tab, iteration)

                # recursively optimize sub-addition
                cur_results = __factorizeMultiTerm(multi_term, cur_usage_tab, index_tab, volatile_tab, iteration, False,level+1)
                (multi_term, cur_usage_tab) = cur_results[0]

                # update usage table (statement table)
                if (def_map.has_key(factor.name)):
                    infos = def_map[factor.name]
                    cur_usage_tab.insertInfos(infos)
                
                if (len(multi_term) == 1):
                    #print '1',multi_term
                    # get the current right term
                    right_term = multi_term[0]
                    
                    # remove unnecessary intermediates out of usage table
                    cur_def_map = cur_usage_tab.removeIntermediates([right_term.optimized_i_arr.name], [right_term.mult])

                    # collect array references
                    cur_arr_set = set()
                    for se in right_term.mult.subexps:
                        cur_arr_set |= set([se.name])
                    
                    # update usage table (statement table)
                    for a in cur_arr_set:
                        if (cur_def_map.has_key(a)):
                            infos = cur_def_map[a]
                            cur_usage_tab.insertInfos(infos)

                    # create the factorized multiplication
                    fact_subexps = [factor] + right_term.mult.subexps
                    fact_mult = Multiplication(fact_subexps, right_term.mult.coef)
                    fact_mult.setOps(sym_groups)

                else:

                    # update usage table (top array references)
                    for t in multi_term:
                        if (isinstance(t, Term)):
                            cur_usage_tab.top_arefs.remove(t.optimized_i_arr.name)
                        else:
                            cur_usage_tab.top_arefs.remove(t.name)

                    # unwrap terms
                    n_multi_term = []
                    for t in multi_term:
                        if (isinstance(t, Term)):
                            n_multi_term.append(t.optimized_i_arr)
                        else:
                            n_multi_term.append(t)
                    multi_term = n_multi_term
                    
                    repr = 0
                    # find representative for addition expression
                    for x in range(0,len(multi_term)):
                        if (multi_term[x].name < multi_term[repr].name): repr = x
                        elif (multi_term[x].name == multi_term[repr].name):
                            #if multi_term[x].name == '_a956': print multi_term
                            if abs(multi_term[x].coef) > abs(multi_term[repr].coef): repr = x
                            #elif abs(multi_term[x].coef) == abs(multi_term[repr].coef):
                                #if multi_term[x].upper_inds[0] > multi_term[repr].upper_inds[0]: repr = x
                    temp=multi_term[0]
                    multi_term[0]=multi_term[repr]
                    multi_term[repr]=temp
                    if (multi_term[0].coef<0):
                        for x in range(0,len(multi_term)):
                            multi_term[x].coef = -multi_term[x].coef
                        factor.multiplyCoef(-1)
                        
                    # create intermediate for the addition
                    add = Addition(multi_term)
                    #print
                    #print '-->add',add
                    
                    add_info = cur_usage_tab.stmt_tab.getInfoForExp(add, index_tab, volatile_tab, iteration)
                    #print
                    #print '-->add2',add
                    #print
                    #print 'add_indo',add_info.lhs
                    add_i_arr = add_info.lhs.replicate()
                    #print 'add_rep',add_i_arr
                    ren_tab = buildRenamingTable(add_info.from_inds, add_info.to_inds)
                    renameIndices(add_i_arr, ren_tab)
                    #print 'add_rename',add_i_arr

                    # create the factorized multiplication
                    
                    fact_mult = Multiplication([factor, add_i_arr])
                    fact_mult.setOps(sym_groups)
                    
                # single-term optimization on the factorized multiplication
                cur_results = single_term_opt.optimize(fact_mult, cur_usage_tab.stmt_tab, index_tab, volatile_tab, iteration)
                (fact_i_arr, fact_info, cur_stmt_tab) = cur_results[0]
                cur_usage_tab.stmt_tab = cur_stmt_tab

                # create factorized term
                fact_term = Term(fact_mult)
                fact_term.optimized_i_arr = fact_i_arr
                fact_term.optimized_info = fact_info
                
                # update usage table (top array references)
                cur_usage_tab.top_arefs.append(fact_i_arr.name)

                # count factorized cost
                all_i_arr_names = cur_usage_tab.stmt_tab.getAllIntermediateNames()
                volatile_list = cur_usage_tab.stmt_tab.getVolatileIntermediateNames(all_i_arr_names)
                updated_volatile_tab = volatile_tab.copy()
                for v in volatile_list:
                    updated_volatile_tab[v] = None
                fact_cost = cur_usage_tab.stmt_tab.getCostForIntermediateNames(cur_usage_tab.stmt_tab.getAllIntermediateNames())
                fact_cost += countOpCost(fact_term.optimized_i_arr, index_tab, updated_volatile_tab, iteration)[0]

                # calculate profit
                cur_profit = unfact_cost - fact_cost
                #print cur_profit
                #print 'unfact',unfact_cost
                #print 'fact',fact_cost

                # store the best results
                if (cur_profit > 0):
                    if (best_profit == None or cur_profit > best_profit):
                        best_results = [(fact_term, cur_usage_tab)]
                        best_profit = cur_profit
                    elif (best_profit == cur_profit):
                        best_results.append((fact_term, cur_usage_tab))
                        
    #for zz in range(0, level):
    #    print'\t',
    #print 'end two term'
    return (best_results, best_profit)

#---------------------------------------------------------------------

def __prepareForRecursion(multi_term, def_map, usage_tab, index_tab, volatile_tab, iteration):

    # copy usage table and volatile table
    usage_tab = usage_tab.shallow_copy()
    volatile_tab = volatile_tab.copy()

    # expand any one-array term into addition
    #xxx
    
    # expand any one-array term into multiplication
    #xxx

    # collect array references
    arr_set = set()
    for t in multi_term:
        if (isinstance(t, Array)):
            arr_set |= set([t.name])
        elif (isinstance(t, Multiplication)):
            for se in t.subexps:
                arr_set |= set([se.name])
        else:
            raise OptimizerError('%s: unexpected type' % __name__)

    # update usage table (statement table)
    for a in arr_set:
        if (def_map.has_key(a)):
            infos = def_map[a]
            usage_tab.insertInfos(infos)

    # update volatile table
    for a in arr_set:
        if (def_map.has_key(a)):
            info = def_map[a][0]
            if (info.is_volatile):
                volatile_tab[a] = None
            assert(a == info.lhs.name), '%s: mismatched LHS array name' % __name__

    # wrap terms
    n_multi_term = []
    for t in multi_term:
        if (isinstance(t, Multiplication)):
            n_multi_term.append(Term(t))
        else:
            n_multi_term.append(t)        
    multi_term = n_multi_term
    
    # collect terms
    terms = filter(lambda x: isinstance(x, Term), multi_term)

    if (__USE_CSE):

        # apply single-term optimization to count weight for each term        
        for t in terms:
            cur_stmt_tab = StmtTable()
            cur_stmt_tab.turnOffCommonSubexp()
            cur_results = single_term_opt.optimize(t.mult, cur_stmt_tab, index_tab, volatile_tab, iteration)
            (cur_i_arr, cur_info, cur_stmt_tab) = cur_results[0]
            t.weight = cur_stmt_tab.getCostForIntermediateNames(cur_stmt_tab.getAllIntermediateNames())
    
        # sort terms from heaviest to lightest
        terms.sort(lambda x,y: -cmp(x.weight, y.weight))
    
        # common-subexpression elimination
        cur_results = __cse(terms, usage_tab.stmt_tab, index_tab, volatile_tab, iteration,
                            __CSE_INTERVAL, __CSE_SOFT_LIMIT, __CSE_HARD_LIMIT, False)
        (cur_i_arrs, cur_infos, cur_stmt_tab) = cur_results[0] 
        usage_tab.stmt_tab = cur_stmt_tab
        for (cur_term, cur_i_arr, cur_info) in zip(terms, cur_i_arrs, cur_infos):
            cur_term.optimized_i_arr = cur_i_arr
            cur_term.optimized_info = cur_info

    else:

        # single-term optimization
        cur_stmt_tab = usage_tab.stmt_tab
        for t in terms:
            cur_results = single_term_opt.optimize(t.mult, cur_stmt_tab, index_tab, volatile_tab, iteration)
            (cur_i_arr, cur_info, cur_stmt_tab) = cur_results[0]
            t.optimized_i_arr = cur_i_arr
            t.optimized_info = cur_info
        usage_tab.stmt_tab = cur_stmt_tab
        
    # add identical arrays
    #xxx
    
    # compute array symbols at each term
    for t in terms:
        (mult, arr_symbols) = __symbolizeMult(t.mult)
        t.mult = mult
        t.arr_symbols = arr_symbols
    
    # update usage table (top array references)
    for t in multi_term:
        if (isinstance(t, Term)):
            usage_tab.top_arefs.append(t.optimized_i_arr.name)
        else:
            usage_tab.top_arefs.append(t.name)
            
    return (multi_term, usage_tab)
   
#---------------------------------------------------------------------

def __countCostForMultiTerm(multi_term, usage_tab, index_tab, volatile_tab, iteration):

    # update volatile table
    volatile_list = usage_tab.stmt_tab.getVolatileIntermediateNames(usage_tab.stmt_tab.getAllIntermediateNames())
    updated_volatile_tab = volatile_tab.copy()
    for v in volatile_list:
        updated_volatile_tab[v] = None

    # create top-level expression
    if (len(multi_term) > 1):
        subexps = []
        for t in multi_term:
            if (isinstance(t, Term)):
                subexps.append(t.optimized_i_arr)
            else:
                subexps.append(t)
        exp = Addition(subexps)
    else:
        exp = multi_term[0].optimized_i_arr

    # calculate operation cost
    cost = usage_tab.stmt_tab.getCostForIntermediateNames(usage_tab.stmt_tab.getAllIntermediateNames())
    cost += countOpCost(exp, index_tab, updated_volatile_tab, iteration)[0]
    
    # return operation cost
    return cost

#---------------------------------------------------------------------

def __symbolizeMult(mult):
    coef = mult.coef
    arrs = []
    symbols = []
    
    for se in mult.subexps:
        (a, sy, s) = __getArraySymbol(se, mult.sum_inds)
        coef *= s
        arrs.append(a)
        symbols.append(sy)
        
    m = Multiplication(arrs, coef)
    m.setOps(mult.sym_groups)

    #print 'm',m
    #print 's',symbols    
    return (m, symbols)
                                                          
def __getArraySymbol(arr, sum_inds):
    r_arr = arr.replicate()
    sign = 1

    inds = r_arr.upper_inds + r_arr.lower_inds
    from_inds = []
    to_inds = []
    for g in r_arr.sym_groups:
        original_inds = filter(lambda x: x in g, inds)
        exts = []
        sums = []
        for i in original_inds:
            if (i in sum_inds):
                sums.append(i)
            else:
                exts.append(i)
        exts.sort()
        ordered_inds = exts + sums
        
        from_inds.extend(original_inds)
        to_inds.extend(ordered_inds)
        sign *= __countParitySign(original_inds, ordered_inds)
        
    ren_tab = buildRenamingTable(from_inds, to_inds)

    renameIndices(r_arr, ren_tab)
    
    inds = r_arr.upper_inds + r_arr.lower_inds
    
    symbol = ''
    symbol += r_arr.name + '['
    for (i, ind) in enumerate(inds):
        if (i > 0):
            symbol += ','
        if (ind in sum_inds):
            symbol += '#i'
        else:
            symbol += str(ind)
    symbol += ']'
    
    return (r_arr, symbol, sign)

#---------------------------------------------------------------------

def __countParitySign(ls1, ls2):
    ls2 = ls2[:]
    parity = 0
    for i in range(len(ls1)):
        if (ls1[i] != ls2[i]):
            swap_i = ls2.index(ls1[i])
            ls2[swap_i] = ls2[i]
            ls2[i] = ls1[i]
            parity += 1
    if (parity % 2 == 0):
        return 1
    else:
        return -1
    

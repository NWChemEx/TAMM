
# For each term (defined as a product of tensors), find out the case where we have symmetric loops. We have a symmetric loop
# for a subset of summation indices if they are of the same type (O or V) and appear together in exactly two tensors.
# Let us say these subsets are of size k1, k2, etc.
# Option 1: For this term, we MULTIPLY it's existing constant by the factor k1! * k2! *
# Option 2: For this term, we DIVIDE it's existing constant by the factor k1! * k2! *

import os
import sys
from antlr4 import *

from OpMinLexer import OpMinLexer
from OpMinParser import OpMinParser
from OpMinVisitor import TAMMtoTAMM, MinimizeTemps

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_stream = FileStream(sys.argv[1])
    else:
        print "Please provide an input file!"
        sys.exit(1)

    mdoption = 0; #default = multiply

    try:
        mdoption = int(sys.argv[2]) # > 0 for divide,
    except:
        pass

    lexer = OpMinLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = OpMinParser(token_stream)
    tree = parser.translation_unit()

    fname = os.path.basename(sys.argv[1])
    fname = fname.split(".")[0]
    fname = fname.split("_")

    methodName = fname[0]
    oplabel = "tamm"

    if len(fname) == 2:
        fname = fname[1]
        if fname.strip():
            oplabel = fname


    ci = 0
    for c in oplabel:
        if c.isdigit(): ci+=1
        else: break

    #oplabel = oplabel[ci:]

    t2tEq = (oplabel + " {\n")

    visitor = TAMMtoTAMM(mdoption)
    single_use_temps, streq = visitor.visitTranslation_unit(tree)

    t2tEq += streq + "}"


    remove_temps = dict()
    for s in single_use_temps:
        if single_use_temps[s]:
            remove_temps[s] = single_use_temps[s]

    tamm_file = os.path.basename(sys.argv[1]).split(".")[0]+'_initial.eq'
    if mdoption: tamm_file = os.path.basename(sys.argv[1]).split(".")[0]+'_prefinal.eq'
    with open(tamm_file, 'w') as tr:
        tr.write(t2tEq)

    if not mdoption: sys.exit(0)

    input_stream = FileStream(tamm_file)
    lexer = OpMinLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = OpMinParser(token_stream)
    tree = parser.translation_unit()

    t2tEq = (oplabel + " {\n")
    visitor = MinimizeTemps(remove_temps)
    eqstr = visitor.visitTranslation_unit(tree)
    t2tEq += eqstr + "}"

    print(t2tEq)
    # tamm_file = os.path.basename(sys.argv[1]).split(".")[0]+'_final.eq'
    # with open(tamm_file, 'w') as tr:
    #     tr.write(t2tEq)


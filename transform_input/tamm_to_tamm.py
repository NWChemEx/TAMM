
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
from OpMinVisitor import TAMMtoTAMM

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

    print(oplabel + " {\n")

    visitor = TAMMtoTAMM(mdoption)
    visitor.visitTranslation_unit(tree)

    print("}")

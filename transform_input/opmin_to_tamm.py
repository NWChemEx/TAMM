
import os
import sys
from antlr4 import *
from antlr4.InputStream import InputStream
from OpMinLexer import OpMinLexer
from OpMinParser import OpMinParser
from OpMinVisitor import OpminOutToTAMM

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_stream = FileStream(sys.argv[1])
    else:
        print "Please provide an input file!"
        sys.exit(1)

    lexer = OpMinLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = OpMinParser(token_stream)
    tree = parser.translation_unit()

    fname = os.path.basename(sys.argv[1])
    fname = fname.split(".")[0]
    fname = fname.rsplit("_",1)

    methodName = fname[0]
    oplabel = fname[0]

    if len(fname) == 2:
        oplabel = fname[1]


    ci = 0
    for c in oplabel:
        if c.isdigit(): ci+=1
        else: break

    oplabel = oplabel[ci:]

    print(oplabel + " {\n")

    visitor = OpminOutToTAMM()
    visitor.visitTranslation_unit(tree)

    print("}")

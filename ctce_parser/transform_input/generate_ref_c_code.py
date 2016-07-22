
import os
import sys
from antlr4 import *
from antlr4.InputStream import InputStream
from OpMinLexer import OpMinLexer
from OpMinParser import OpMinParser
from OpMinVisitor import OpMinVisitor

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
    fname = os.path.splitext(fname)[0]
    fname = fname.split("_")

    if (len(fname) != 2):
        print "File name should be of the form ccsd_t1.eq"
        sys.exit(1)

    visitor = OpMinVisitor(fname[0],fname[1])
    visitor.visit(tree)





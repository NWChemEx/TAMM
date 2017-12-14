import os
import sys
from antlr4 import *

genpath = os.path.abspath(os.getcwd()+"/../gen/")
sys.path.append(genpath)

from NWXLexer import NWXLexer
from NWXParser import NWXParser
from NWXVisitor import NWXVisitor

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_stream = FileStream(sys.argv[1])
    else:
        print "Please provide an input file!"
        sys.exit(1)

    lexer = NWXLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = NWXParser(token_stream)
    tree = parser.nwchem_input()

    visitor = NWXVisitor()
    visitor.visitNwchem_input(tree)
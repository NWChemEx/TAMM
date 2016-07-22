
import sys
from antlr4 import *
from antlr4.InputStream import InputStream
from OpMinLexer import OpMinLexer
from OpMinParser import OpMinParser
from OpMinVisitor import OpminOutToCTCE

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

    visitor = OpminOutToCTCE()
    visitor.visitTranslation_unit(tree)



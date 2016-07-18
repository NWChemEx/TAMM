
import sys
from antlr4 import *
from antlr4.InputStream import InputStream
from SoTCELexer import SoTCELexer
from SoTCEParser import SoTCEParser
from SoTCEVisitor import SoTCEVisitor

def print_index_decls(res):
    hind = res[0]
    pind = res[1]

    his = "index "
    for h in range(1,hind+1):
        his += "h" + str(h) + ","

    his = his[:-1]
    his += " = O;\nindex "
    for p in range(1,pind+1):
        his += "p" + str(p) + ","

    his = his[:-1]
    his += " = V;\n"

    print his


if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_stream = FileStream(sys.argv[1])
    else:
        input_stream = InputStream(sys.stdin.readline())

    lexer = SoTCELexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = SoTCEParser(token_stream)
    tree = parser.translation_unit()

    print "{\n"

    visitor = SoTCEVisitor()
    res = visitor.visitTranslation_unit(tree)

    ad = res[0]
    for a in ad:
        print a

    print res[1]

    print "\n}"

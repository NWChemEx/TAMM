
import sys
from antlr4 import *
from antlr4.InputStream import InputStream
from OpMinLexer import OpMinLexer
from OpMinParser import OpMinParser
from OpMinVisitor import OpMinVisitor
from OpMinVisitor import OpminOutToCTCE

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

    lexer = OpMinLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = OpMinParser(token_stream)
    tree = parser.translation_unit()


    gen_ctce_output = 0
    try:
        gen_ctce_output = sys.argv[2]

    except: pass

    if gen_ctce_output==0:
        visitor = OpMinVisitor()
        visitor.visit(tree)

    else:
        visitor = OpminOutToCTCE()
        visitor.visit(tree)



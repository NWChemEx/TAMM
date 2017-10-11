
import sys
from antlr4 import *
from antlr4.InputStream import InputStream
from SoTCELexer import SoTCELexer
from SoTCEParser import SoTCEParser
from SoTCEVisitor import SoTCEVisitor

def print_index_decls(res):
    pind = res[0]
    hind = res[1]

    print "range O = 10;"
    print "range V = 100;"
    print "range N = 110;\n"

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


    visitor = SoTCEVisitor()
    res = visitor.visitTranslation_unit(tree)

    print "{\n"

    print_index_decls(res[3])
    ad = res[0]
    for a in ad:
        print a

    ltype = []
    for r in res[2]:
        if r[0] == 'h':
            if r[-1] == "*": ltype.append('O*')
            else: ltype.append('O')
        elif r[0] == 'p':
            if r[-1] == "*": ltype.append('V*')
            else: ltype.append('V')

    upper = ltype[0:len(ltype) / 2]
    lower = ltype[len(ltype) / 2:]

    newup = []
    newlow = []
    for u in range(0, len(upper)):
        if upper[u][-1] != "*": newup.append(upper[u])

    for u in range(0, len(lower)):
        if lower[u][-1] != "*": newlow.append(lower[u])

    upper = ",".join(newup)
    lower = ",".join(newlow)

    idims = []
    for r in res[2]:
        if r[-1]!="*": idims.append(r)

    print "array i0([" + upper + "][" + lower + "]);"
    if idims: print "\ni0[" + ",".join(idims) + "] = " + res[1] + ";"
    else: print "\ni0 = " + res[1] + ";"

    print "\n}"

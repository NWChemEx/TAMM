
import os
import sys
from antlr4 import *
from antlr4.InputStream import InputStream
from NWChemTCELexer import NWChemTCELexer
from NWChemTCEParser import NWChemTCEParser
from NWChemTCEVisitor import NWChemTCEVisitor
from NWChemTCEVisitor import NWChemTCEVisitorExecOrder

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
        print "Please provide an input file!"
        sys.exit(1)

    lexer = NWChemTCELexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = NWChemTCEParser(token_stream)
    tree = parser.translation_unit()

    fname = os.path.basename(sys.argv[1])
    fname = fname.split(".")[0]
    fname = fname.split("_")

    if (len(fname) != 2):
        print "File name should be of the form ccsd_t1.eq"
        sys.exit(1)

    visitor = NWChemTCEVisitorExecOrder()
    res = visitor.visitTranslation_unit(tree)

    exec_order = res[0]
    array_decls = res[1]

    visitor = NWChemTCEVisitor(fname[0],fname[1])
    for stmt in exec_order:
        visitor.visitStatement(stmt)

    code_idecls = visitor.getCode()


    print fname[1] + " {\n"

    print_index_decls(res[2])

    for arr in array_decls:
        print(arr)

    for arr in code_idecls[1].keys():
        if (arr != "i0"): print "array " + arr + code_idecls[1][arr]

    print ""
    print code_idecls[0]

    print "}"

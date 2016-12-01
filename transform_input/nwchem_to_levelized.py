
import os
import re
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
        input_file = sys.argv[1]
    else:
        print "Please provide an input file!"
        sys.exit(1)


    delete_index_list = []
    cur_kbn = ""
    max_i = 0

    tmpInputFile = open(input_file+".tmp",'w')
    #Check for indices to be deleted.
    inpFile = open(input_file,'r')
    for lno, line in enumerate(inpFile, 1):

        #line = line.lstrip()
        if line.startswith("kbn"):
            line = line.strip()
            line = line.replace(";", "")

            dindex = line.split(" ")
            if len(dindex) == 3:
                assert (dindex[2] == 'm')
            #delete_index_list.append(dindex[1])
            cur_kbn = dindex[1]
        else:
            if cur_kbn:
                #line = line.replace(cur_kbn, cur_kbn+"*")
                line = re.sub(r'\b' + cur_kbn + r'\b', cur_kbn+"*", line)
                cur_kbn = ""
            cline = line.strip()
            if not cline.startswith("i"):
                if cline:
                    print "Error at Line " + str(lno) + ". Line cannot start with anything other than kbn or i[1-9]+\n"
                    sys.exit(2)
            else:
                geti = cline.split(" ")[0]
                geti = int(geti[1:])
                max_i = max(max_i,geti)

            tmpInputFile.write(line)

    inpFile.close()
    tmpInputFile.close()
    #print "max i = " + str(max_i)

    input_stream = FileStream(input_file+".tmp")

    lexer = NWChemTCELexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = NWChemTCEParser(token_stream)
    tree = parser.translation_unit()

    fname = os.path.basename(sys.argv[1])
    fname = fname.split(".")[0]
    fname = fname.split("_",1)

    methodName = fname[0]
    oplabel = fname[0]

    if len(fname) == 2:
        oplabel = fname[1]


    ci = 0
    for c in oplabel:
        if c.isdigit(): ci+=1
        else: break

    oplabel = oplabel[ci:]

    # if (len(fname) != 2):
    #     print "File name should be of the form ccsd_t1.eq"
    #     sys.exit(1)


    visitor = NWChemTCEVisitorExecOrder(max_i,methodName,oplabel)
    res = visitor.visitTranslation_unit(tree)

    exec_order = res[0]
    array_decls = res[1]

    visitor = NWChemTCEVisitor(methodName,oplabel)
    for stmt in exec_order:
        visitor.visitStatement(stmt)

    code_idecls = visitor.getCode()


    print oplabel + " {\n"

    print_index_decls(res[2])

    for arr in array_decls:
        print(arr)

    for arr in code_idecls[1].keys():
        if (arr != "i0"): print "array " + arr + code_idecls[1][arr]

    print ""
    print code_idecls[0]

    print "}"

    os.remove(input_file+".tmp")

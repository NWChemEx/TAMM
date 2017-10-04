
import os
import sys
from antlr4 import *
from antlr4.InputStream import InputStream
from OpMinLexer import OpMinLexer
from OpMinParser import OpMinParser
from OpMinVisitor import OpminOutToTAMM, OpminTAMMSplitAdds

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

    op2tammstr = oplabel + " {\n"

    visitor = OpminOutToTAMM()
    op2tammstr += visitor.visitTranslation_unit(tree)
    op2tammstr += "}"
    #print(op2tammstr)

    tamm_file = os.path.basename(sys.argv[1]).split(".")[0]+'_2tamm.eq'
    with open(tamm_file, 'w') as tr:
        tr.write(op2tammstr)

    input_stream = FileStream(tamm_file)
    lexer = OpMinLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = OpMinParser(token_stream)
    tree = parser.translation_unit()

    op2tammstr = (oplabel + " {\n")
    visitor = OpminTAMMSplitAdds()
    eqstr = visitor.visitTranslation_unit(tree)
    op2tammstr += eqstr + "}"

    tamm_file = os.path.basename(sys.argv[1]).split(".")[0]+'_splitAdds.eq'
    with open(tamm_file, 'w') as tr:
        tr.write(op2tammstr)




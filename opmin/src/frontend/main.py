import const_folder
import error
import expander
import flattener
import parser
import semant
import translator


def main(ifname):

    # initialize error handler
    error.setFilename(ifname)

    # open and read input file
    f = open(ifname, 'r')
    source_text = f.read()
    f.close()

    # parse
    p = parser.createParser()
    trans_unit = p.parse(source_text)

    # semantic check
    semant.semantCheck(trans_unit)

    # expand intermediate arrays
    expander.expand(trans_unit)

    # flatten expression
    flattener.flatten(trans_unit)

    # fold constants
    const_folder.fold(trans_unit)

    # semantic check
    semant.semantCheck(trans_unit)

    # translate
    normalized_trans_unit = translator.translate(trans_unit)

    # symmetry check
    semant.symCheck(normalized_trans_unit)

    # apply symmetrization operators
    translator.applySym(normalized_trans_unit)

    # close error handler
    error.clearFilename()

    return normalized_trans_unit

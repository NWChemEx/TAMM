import unparser
import tester
from time import time


def main(original_trans_unit, optimized_trans_unit, ofname):

    # unparsing
    output_text = unparser.unparse(optimized_trans_unit)

    # opening and writing output file
    f = open(ofname, 'w')
    f.write(output_text)
    f.close()

    # testing correctness
    print '--- Start testing ---'
    t1 = time()
    output_correct = tester.test(original_trans_unit, optimized_trans_unit)
    t2 = time()
    print '--> time for testing correctness: %s secs' % (t2-t1)

    if (output_correct):
        print '--> the optimized result is CORRECT'
    else:
        print '--> the optimized result is INCORRECT'
    print '--- Finished testing ---'

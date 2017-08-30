import optparse
import os
import sys

import frontend.main
import optimizer.main
import backend.main


def main():
    print '****** OPMIN starts ******'
    print ''

    (ifnames, use_cse, use_fact, use_refine) = __readCommandLineArguments()
    ofnames = __createOutputFilenames(ifnames)

    for ifname, ofname in zip(ifnames, ofnames):
        print '=== Processing file: "%s" ===' % ifname

        # frontend
        print '--- Start input reading and pre-processing ---'
        original_trans_unit = frontend.main.main(ifname)
        print '--- Finished input reading and pre-processing ---'

        print '--- Unoptimized equation ---'

        # optimizer
        print '--- Start optimization ---'
        optimized_trans_unit = optimizer.main.main(original_trans_unit.replicate(), use_cse, use_fact, use_refine)
        print '--- Finished optimization ---'

        # print '--- Optimized equation ---'
        # print str(optimized_trans_unit)

        # backend
        print '--- Start output writing ---'
        backend.main.main(original_trans_unit, optimized_trans_unit, ofname)
        print '--- Finished output writing ---'

        print ''

    print '****** OPMIN finishes ******'


def __readCommandLineArguments():
    cmd_parser = __createCmdParser()
    (options, args) = cmd_parser.parse_args()

    ifnames = args
    if (len(ifnames) == 0):
        print 'error: no input files defined'
        sys.exit(-1)

    for ifname in ifnames:
        try:
            f = open(ifname, 'r')
            f.close()
        except:
            print 'error: cannot open input file: %s' % ifname
            sys.exit(-1)

    use_cse = options.use_cse
    use_fact = options.use_fact
    use_refine = options.use_refine

    return (ifnames, use_cse, use_fact, use_refine)


def __createOutputFilenames(ifnames):
    ofnames = []
    for i in ifnames:
        (header, trailer) = os.path.split(i)
        o = os.path.join(header, trailer + '.out')
        try:
            f = open(o, 'w')
            f.close()
        except:
            print 'error: cannot open output file: %s' % o
            sys.exit(-1)
        ofnames.append(o)
    return ofnames


def __createCmdParser():
    usage = 'usage: %prog [options] INFILE...'
    description = 'description: Compile shell for operation minimization'
    p = optparse.OptionParser(usage=usage, description=description)
    p.add_option('-c', '--commonsubexp', help='turn on common subexpression elimination',
                 action='store_true', dest='use_cse')
    p.add_option('-f', '--factorize', help='turn on factorization', action='store_true', dest='use_fact')
    p.add_option('-r', '--refinement', help='do factorization first and then refine the results using cse and fact',
                 action='store_true', dest='use_refine')
    return p

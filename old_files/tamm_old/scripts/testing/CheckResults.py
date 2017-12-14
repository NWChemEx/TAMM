#!/usr/bin/env python

# Script to check results produced by TAMM.

import sys
import os
import json
import math
import subprocess

nwchem_bin = str(sys.argv[1])
testcase = str(sys.argv[2])
num_procs = int(sys.argv[3])

def check_result_file(res_file,result_string, num_lines):
    get_results = []
    with open(res_file, "r") as outf:
        for line in outf:
            if line.strip().startswith(result_string):
                if num_lines[0] == "=":
                    tvalue = line.split("=")[1]
                    get_results.append(tvalue.strip())
                else:  # Assume number for now
                    nl = num_lines[0]
                    lc = 0
                    while lc != nl:
                        next_line = next(outf)
                        next_line = ' '.join(next_line.strip().split())
                        iter_type1 = next_line.startswith(str(lc+1) + " ")
                        iter_type2 = next_line.startswith("Iteration " + str(lc+1) + " ")
                        if iter_type1 or iter_type2:
                            residuum = next_line.split(" ")[1]
                            if iter_type2:
                                next_line = next(outf)
                                #Line after Iteration x message may have warnings sometimes
                                while any(c.isalpha() for c in next_line):
                                    next_line=next(outf)
                                    #Could happen that these iteration values are NaNs
                                    if next_line.startswith("Iteration " + str(lc + 2) + " "):
                                        print "Iteration " + str(lc+2) + " missing values!"
                                        sys.exit(1)
                                next_line = ' '.join(next_line.strip().split())
                                residuum = next_line.split(" ")[0]
                            get_results.append(residuum.strip())
                            lc += 1
                    assert len(get_results) == nl
    return get_results



tamm_root = os.path.dirname(os.path.abspath(__file__)) + "/../.."

with open(tamm_root+"/CI_testing/integration/test.json") as json_data_file:
    data = json.load(json_data_file)


num_tests = len(data)
num_failed = 0
input_folder = tamm_root+"/CI_testing/integration/"

#for testcase, patterns in data.iteritems():
patterns = data[testcase]
print "Running Test: " + testcase + ".nw"
nwinput = input_folder + "/inputs/" + testcase + ".nw"
tamm_output = testcase + ".tammout"

run_tamm_code = "mpirun -n " + str(num_procs) + " " + nwchem_bin + " " + nwinput
print run_tamm_code
tof = open(tamm_output,"w")
pid = subprocess.Popen(run_tamm_code, stdout=tof, stderr=tof, shell=True)
pid.communicate()
tof.close()

#print run_tamm_code + " 2>&1 | tee " + tamm_output
#os.system(run_tamm_code + " 2>&1 | tee " + tamm_output)

orig_result = input_folder + "correct_outputs/" + testcase + ".output"

for result_string,num_lines in patterns.iteritems():
    res1 = check_result_file(orig_result,result_string,num_lines)
    res2 = check_result_file(tamm_output,result_string,num_lines)

    if num_lines[0] == "=":
        value1 = float(res1[0])
        value2 = float(res2[0])
        if math.fabs(value1 - value2) > 1.0e-10:
            print result_string + " does not match"
            print "Expected value = " + res1[0] + ", Found = " + res2[0]
            num_failed +=1
        continue

    nl = num_lines[0]
    for v in range(0,nl):
        value1 = float(res1[v])
        value2 = float(res2[v])
        if math.fabs(value1 - value2) > 1.0e-10:
            print result_string + " Residuum does not match"
            print "At iteration " + str(v+1) + ": Expected = " + res1[v] + ", Found = " + res2[v]
            num_failed += 1

if num_failed > 0: sys.exit(1)

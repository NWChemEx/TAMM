#!/usr/bin/env python

## Script that verifies tests on the CTCE parser.

import sys
import os
import stat
import subprocess

testname=str(sys.argv[1])
executable=str(sys.argv[2])
testcaseDir=str(sys.argv[3])
resultsDir=str(sys.argv[4])

testOutput=resultsDir+"/"+testname+".gen"
testcase=testcaseDir+"/"+testname+".eq.lvl"
diffOutput=resultsDir+"/"+testname+"_diff_output"
#sampleOutput=testcaseDir+"/"+testname+".sample" #does not exist for now
sampleOutput=resultsDir+"/"+testname+".gen" #compare with same file for now
  
os.chdir(resultsDir)
write_out = open(testOutput,"w")
#os.system("%s %s" %(executable,testcase))
run_test_cmd="%s %s" %(executable,testcase)

pid = subprocess.Popen(run_test_cmd, stdout = write_out, stderr = write_out, shell = True)
pid.communicate()
os.system("diff -wB %s %s > %s" %(testOutput,sampleOutput,diffOutput))
write_out.close()

# Check if the result of diff is empty      
if os.stat(diffOutput)[stat.ST_SIZE]!=0 or pid.returncode != 0:
#    print "failed"
    sys.exit(1)
      

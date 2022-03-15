#!/usr/bin/env python

import sys
import os
import json
import math

def isclose(a, b, rel_tol=1e-09, abs_tol=0):
  return abs(a-b) <= rel_tol #max(rel_tol * max(abs(a), abs(b)), abs_tol)

if len(sys.argv) < 2:
    print("\nUsage: python3 compare_results.py reference_results_path current_results_path")
    sys.exit(1)

ref_res_path = os.path.abspath(str(sys.argv[1]))
cur_res_path = os.path.abspath(str(sys.argv[2]))

#check if above paths exist
if not os.path.exists(ref_res_path): 
    print("ERROR: " + ref_res_path + " does not exist!")
    sys.exit(1)

if not os.path.exists(cur_res_path): 
    print("ERROR: " + cur_res_path + " does not exist!")
    sys.exit(1)

ref_files = os.listdir(ref_res_path)
cur_files = os.listdir(cur_res_path)


def check_pt_results(ref_energy,cur_energy,ccsd_threshold,en_str):
    if (not isclose(ref_energy, cur_energy, ccsd_threshold)):
        print("ERROR: mismatch in " + en_str + "\nreference: " + str(ref_energy) +
        ", current: " + str(cur_energy))
        sys.exit(1) 

for ref_file in ref_files:
    if ref_file not in cur_files:
        print("ERROR: " + ref_file + " not available in " + cur_res_path)
        sys.exit(1)
    
    with open(ref_res_path+"/"+ref_file) as ref_json_file:
        ref_data = json.load(ref_json_file)

    with open(cur_res_path+"/"+ref_file) as cur_json_file:
        cur_data = json.load(cur_json_file)    

    ref_scf_energy = ref_data["output"]["SCF"]["final_energy"]
    cur_scf_energy = cur_data["output"]["SCF"]["final_energy"]

    ref_ccsd_energy = ref_data["output"]["CCSD"]["final_energy"]["correlation"]
    cur_ccsd_energy = cur_data["output"]["CCSD"]["final_energy"]["correlation"]

    if not isclose(ref_scf_energy, cur_scf_energy):
        print("ERROR: SCF energy does not match. reference: " + str(ref_scf_energy) + ", current: " + str(cur_scf_energy))
        sys.exit(1)

    ccsd_threshold = ref_data["input"]["CCSD"]["threshold"]
    if not isclose(ref_ccsd_energy, cur_ccsd_energy, ccsd_threshold):
        print("ERROR: CCSD correlation energy does not match. reference: " + str(ref_ccsd_energy) + ", current: " + str(cur_ccsd_energy))
        sys.exit(1)

    ref_pt_data = ref_data["output"]["CCSD(T)"]
    cur_pt_data = cur_data["output"]["CCSD(T)"]


    ref_correction = ref_pt_data["[T]Energies"]["correction"]
    cur_correction = cur_pt_data["[T]Energies"]["correction"]

    ref_correlation = ref_pt_data["[T]Energies"]["correlation"]
    cur_correlation = cur_pt_data["[T]Energies"]["correlation"]

    ref_total = ref_pt_data["[T]Energies"]["total"]
    cur_total = cur_pt_data["[T]Energies"]["total"]        

    check_pt_results(ref_correction,cur_correction,ccsd_threshold,"[T] Correction Energy")
    check_pt_results(ref_correlation,cur_correlation,ccsd_threshold,"[T] Correlation Energy")
    check_pt_results(ref_total,cur_total,ccsd_threshold,"[T] Total Energy")

    ref_correction = ref_pt_data["(T)Energies"]["correction"]
    cur_correction = cur_pt_data["(T)Energies"]["correction"]

    ref_correlation = ref_pt_data["(T)Energies"]["correlation"]
    cur_correlation = cur_pt_data["(T)Energies"]["correlation"]

    ref_total = ref_pt_data["(T)Energies"]["total"]
    cur_total = cur_pt_data["(T)Energies"]["total"]           

    check_pt_results(ref_correction,cur_correction,ccsd_threshold,"[T] Correction Energy")
    check_pt_results(ref_correlation,cur_correlation,ccsd_threshold,"[T] Correlation Energy")
    check_pt_results(ref_total,cur_total,ccsd_threshold,"[T] Total Energy")

#!/bin/bash 
#[ $# -ge 1 -a -f "$1" ] && input="$1" || input="-"
#[ $# -ge 2 -a -f "$2" ] && input="$2" || input="-" 
#This script check the correctness of generated output 
#  with a reference golden output

# Input parameters to this script
echo $1
test_command=$1
golden_output_file=$2
#eval test_output_file=\$\($test_command\)
test_output_file="$($test_command)"
key_string="CCSD iterations"
total_enery_digits="-b-14"
correlation_enery_digits="-b-13"
residuum_digits="-b-10"
#end of input parameters
#set EXIT ERROR CODE 
if [ -z "$3" ]; then
    head_cut=22
else
    head_cut="$(cat $3)"
fi
error_code=0

#CHECK on total energy
correct_total_energy="$(cat $golden_output_file | grep -A28 "$key_string" | grep -v DIIS | grep total | cut -d "=" -f2 | tr -d " " | cut $total_enery_digits | less)"
test_total_energy="$(echo "$test_output_file" | grep -A28 "$key_string" | grep -v DIIS | grep total | cut -d "=" -f2 | tr -d " " | cut $total_enery_digits | less)"

DIFF_TOTAL_ENERGY=$(diff <(echo "$correct_total_energy") <(echo "$test_total_energy") )
#echo "diff total energy=$DIFF_TOTAL_ENERGY"

if [ "$DIFF_TOTAL_ENERGY" != "" ] 
then
  echo " ***** ERROR ***** TOTAL ENERGY output differs"
	echo "correct_total_energy=$correct_total_energy"
	echo "test_total_energy=$test_total_energy"
    error_code=1
fi

#CHECK on correlation energy
correct_correlation_energy="$(cat $golden_output_file | grep -A28 "$key_string" | grep -v DIIS | grep correlation | cut -d "=" -f2 | tr -d " " | cut $correlation_enery_digits | less)"
test_correlation_energy="$(echo "$test_output_file" | grep -A28 "$key_string" | grep -v DIIS | grep correlation | cut -d "=" -f2 | tr -d " " | cut $correlation_enery_digits | less)"

DIFF_CORRELATION_ENERGY=$(diff <(echo "$correct_correlation_energy") <(echo "$test_correlation_energy") )
#echo "diff correlation energy=$DIFF_CORRELATION_ENERGY"

if [ "$DIFF_CORRELATION_ENERGY" != "" ] 
then
  echo " ***** ERROR ***** CORRELATION ENERGY output differs"
	echo "correct_correlation_energy=$correct_correlation_energy"
	echo "test_correlation_energy=$test_correlation_energy"
    error_code=1
fi

#CHECK on Residuum printed over 
correct_residuum="$(cat $golden_output_file | grep -A28 "$key_string" | grep -v DIIS | head -$head_cut | tail -18 | tr -s " " | cut -d " " -f3 | cut $residuum_digits | less)"
#echo -e "correct_residuum=\n$correct_residuum"
test_residuum="$(echo "$test_output_file" | grep -A28 "$key_string" | grep -v DIIS | head -$head_cut | tail -18 | tr -s " " | cut -d " " -f3 | cut $residuum_digits | less)"
#echo -e "test_residuum=\n$test_residuum"

DIFF_RESIDUUM=$(diff <(echo "$correct_residuum") <(echo "$test_residuum") )
#echo "diff residuum= $DIFF_RESIDUUM"

if [ "$DIFF_RESIDUUM" != "" ]
then
  echo " ***** ERROR ***** residuum output differs"
  echo "correct_residuum=$correct_residuum"
	echo "test_residuum=$test_residuum"
    error_code=1
fi

#Print correctness check comments
if [ $error_code != 1 ]
then
  echo "  Congratulations!! The RESIDUUM output is CORRECT"
  echo "  Congratulations!! The CORRELATION ENERGY output is CORRECT"
  echo "  Congratulations!! The TOTAL ENERGY output is CORRECT"
fi

exit $error_code
#End of script

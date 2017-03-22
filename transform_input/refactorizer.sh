#!/bin/bash

input_file_folder=$1

num_files=0
if [[ -d $input_file_folder ]]; then
    for eqfile in "$input_file_folder"/*.eq
	do
	    allfiles[ $num_files ]="$eqfile"
	    (( num_files++ ))
    	done
elif [[ -f $input_file_folder ]]; then
    allfiles[ $num_files ]="$input_file_folder"  
else
    echo "$input_file_folder is not a valid file or folder!!"
    exit 1
fi

printf "%s\n" "${allfiles[@]}"


for eqfile in "${allfiles[@]}"
do
	eqname=$(basename $eqfile)
	eqname="${eqname%.*}"

	python tamm_to_tamm.py $eqfile 0 $2 $3
	python unfactorize.py "$eqname"_initial.eq > "$eqname"_initial_un.eq
	python ../opmin/src/opmin.py "$eqname"_initial_un.eq
	mv "$eqname"_initial_un.eq.out "$eqname"_opmin_generated.eq
	python opmin_to_tamm.py "$eqname"_opmin_generated.eq 
	python tamm_to_tamm.py "$eqname"_opmin_generated_splitAdds.eq 1 > "$eqname"_final.eq 
	echo "---------------------------------------------------"
	echo " Refactorized file = " "$eqname"_final.eq       
	echo "---------------------------------------------------"
done

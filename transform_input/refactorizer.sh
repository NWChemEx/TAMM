#!/bin/bash
eqname=$(basename $1)
eqname="${eqname%.*}"

python tamm_to_tamm.py $1
python unfactorize.py "$eqname"_initial.eq > "$eqname"_initial_un.eq
python ../opmin/src/opmin.py "$eqname"_initial_un.eq
mv "$eqname"_initial_un.eq.out "$eqname"_opmin_generated.eq
python opmin_to_tamm.py "$eqname"_opmin_generated.eq 
python tamm_to_tamm.py "$eqname"_opmin_generated_splitAdds.eq 1 > "$eqname"_final.eq 
echo "---------------------------------------------------"
echo " Refactorized file = " "$eqname"_final.eq       
echo "---------------------------------------------------"


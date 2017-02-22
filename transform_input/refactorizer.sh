#!/bin/bash
eqname=$(basename $1)
eqname="${eqname%.*}"

python tamm_to_tamm.py $1  > "$eqname"_initial.eq  
python unfactorize.py "$eqname"_initial.eq > "$eqname"_initial_un.eq
python ../opmin/src/opmin.py "$eqname"_initial_un.eq
python opmin_to_tamm.py "$eqname"_initial_un.eq.out 
python opmin_to_tamm.py "$eqname"_initial_un.eq.out > "$eqname"_prefinal.eq
python tamm_to_tamm.py "$eqname"_prefinal.eq 1 > "$eqname"_final.eq 



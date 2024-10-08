#!/bin/bash

# binding for 1 PRs (NICs) using GA_PACKED
if [ $PALS_LOCAL_RANKID -eq 0 ]; then export ZE_AFFINITY_MASK=0
elif [ $PALS_LOCAL_RANKID -eq 1 ]; then export ZE_AFFINITY_MASK=1
elif [ $PALS_LOCAL_RANKID -eq 2 ]; then export ZE_AFFINITY_MASK=2
elif [ $PALS_LOCAL_RANKID -eq 3 ]; then export ZE_AFFINITY_MASK=3
elif [ $PALS_LOCAL_RANKID -eq 4 ]; then export ZE_AFFINITY_MASK=4
elif [ $PALS_LOCAL_RANKID -eq 5 ]; then export ZE_AFFINITY_MASK=5
elif [ $PALS_LOCAL_RANKID -eq 6 ]; then export ZE_AFFINITY_MASK=6
elif [ $PALS_LOCAL_RANKID -eq 7 ]; then export ZE_AFFINITY_MASK=7
elif [ $PALS_LOCAL_RANKID -eq 8 ]; then export ZE_AFFINITY_MASK=8
elif [ $PALS_LOCAL_RANKID -eq 9 ]; then export ZE_AFFINITY_MASK=9
elif [ $PALS_LOCAL_RANKID -eq 10 ]; then export ZE_AFFINITY_MASK=10
elif [ $PALS_LOCAL_RANKID -eq 11 ]; then export ZE_AFFINITY_MASK=11
fi

# Launch the executable:
$*

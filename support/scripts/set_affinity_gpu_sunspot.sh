#!/bin/bash

GPU_ID=$(( PALS_LOCAL_RANKID /  2 ))
TILE_ID=$(( PALS_LOCAL_RANKID % 2 ))

export ZE_AFFINITY_MASK=$GPU_ID.$TILE_ID

if [ $PALS_LOCAL_RANKID -eq 12 ]; then
    export ZE_AFFINITY_MASK=0.0
fi

echo "[I am rank $PMIX_RANK on node `hostname`]  Localrank=$PALS_LOCAL_RANKID, ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK"

# Launch the executable:
$*

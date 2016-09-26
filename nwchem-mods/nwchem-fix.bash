if [ $# -gt 0 ]; then
    cp template.patch gnumake.patch
    sed -i -- "s:(LIB_TEMP):$1:g" ./gnumake.patch
    patch -s -N -r - $NWCHEM_TOP/src/GNUmakefile ./gnumake.patch
    rm gnumake.patch
else
    echo "Usage $0 INSTALL_LIB_PATH"
fi
cp *.F $NWCHEM_TOP/src/tce/

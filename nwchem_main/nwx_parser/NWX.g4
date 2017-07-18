//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//------------------------------------------------------------------------------

grammar NWX;


// TOKENS

// Reserved Keywords
START   :   'START' | 'start';
STOP    :   'STOP' | 'stop';
RESTART  :   'RESTART' | 'restart';
PERMANENT_DIR  :   'PERMANENT_DIR' | 'permanent_dir';
SCRATCH_DIR :   'SCRATCH_DIR' | 'scratch_dir';
MEMORY: 'MEMORY' | 'memory';
ECHO: 'ECHO' | 'echo';
TITLE: 'TITLE' | 'title';
SET: 'SET' | 'set';
UNSET: 'UNSET' | 'unset';
PRINT: 'PRINT' | 'print';
NOPRINT: 'NOPRINT' | 'noprint';
TASK: 'TASK' | 'task';
ECCE_PRINT: 'ECCE_PRINT' | 'ecce_print';

GEOMETRY: 'GEOMETRY' | 'geometry';
SYMMETRY: 'SYMMETRY' | 'symmetry';
LOAD: 'LOAD' | 'load';
RTDB: 'RTDB' | 'rtdb';
CHARGE: 'CHARGE' | 'charge';
UNITS: 'UNITS' | 'units';

COLON: ':';
STAR: '*';
// Identifier
//ID
//    :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
//    ;

//Can be file name, dir path or any other name
ID
        :('a'..'z'|'A'..'Z'|'0'..'9'|'_'|'/'|'.'|'-');
         //| '"' ID '"';

//DIR_FILE_PATH
//    : ('a'..'z'|'A'..'Z'|'0'..'9'|'_'|'/'|'.'|'-')+
//    ;


// Integer Constant
ICONST
    :   '0'..'9'+
    ;

FRAC
    :   ('1'..'9')+ '/' ('1'..'9')+
    ;

// Foalting Point Constant
FCONST
    :   ('0'..'9')+ '.' ('0'..'9')* EXPONENT?
    |   '.' ('0'..'9')+ EXPONENT?
    |   ('0'..'9')+ EXPONENT
    ;

fragment EXPONENT
    :   ('e'|'E') ('+'|'-')? ('0'..'9')+
    ;


// translation-unit
nwchem_input: directive_list EOF;

directive_list: (
                start_directive
                );


/*   memory [[total] <total_size>] [stack <stack_size>]
            [heap <heap_size>] [global <global_size>]
            [<units>] [verify|noverify] [hardfail|nohardfail]
*/

TOTAL: 'TOTAL' | 'total';
STACK: 'STACK' | 'stack';
HEAP: 'HEAP' | 'heap';
GLOBAL: 'GLOBAL' | 'global';
VERIFY: 'VERIFY' | 'verify';
NOVERIFY: 'NOVERIFY' | 'noverify';
HARDFAIL: 'HARDFAIL' | 'hardfail';
NOHARDFAIL: 'NOHARDFAIL' | 'nohardfail';

memory_directive: MEMORY (TOTAL? ICONST)? (STACK ICONST)?
                  (HEAP ICONST)? (GLOBAL ICONST)? ID?
                  (VERIFY | NOVERIFY)? (HARDFAIL | NOHARDFAIL)?;

title_directive: TITLE ID;

NONE: 'NONE' | 'none';
LOW: 'LOW' | 'low';
MEDIUM: 'MEDIUM' | 'medium';
HIGH: 'HIGH' | 'high';
DEBUG: 'DEBUG' | 'debug';

print_directive: ( (PRINT (NONE | LOW | MEDIUM | HIGH | DEBUG)?) | NOPRINT) ID+;

set_directive: SET ID? range_spec ICONST;

unset_directive: UNSET ID STAR?;

/*
(RESTART || START) [<string file_prefix default input_file_prefix>] \
                   [rtdb <string rtdb_file_name default file_prefix.db>]
*/
start_directive:  (START | RESTART) ID? (RTDB ID?)? ;

ecce_print_directive: ECCE_PRINT ID;

permanent_dir_directive: PERMANENT_DIR ( ( (ID | ICONST) COLON)? ID)+;

scratch_dir_directive: SCRATCH_DIR ( ( (ID | ICONST) COLON)? ID)+;

SETTING: 'SETTING' | 'setting';
GROUP: 'GROUP' | 'group';

// [SYMMETRY [group] <string group_name>|<integer group number> [setting <integer setting>] [print] \
//          [tol <real tol default 1d-2>]]

symmetry_directive: SYMMETRY  GROUP (ID | ICONST) (SETTING ICONST)? PRINT? (TOL FCONST)?;

/*
   [<string tag> <real x y z> [vx vy vz] [charge <real charge>] \
          [mass <real mass>] \
          [(nuc || nucl || nucleus) <string nucmodel>]
   ... ]
*/

MASS: 'MASS' | 'mass';

cartesian_coord_directive: (ID FCONST FCONST FCONST (FCONST FCONST FCONST)?
                           (CHARGE FCONST)? (MASS FCONST)? (NUCLEUS ID)?)+;


/* [ LOAD [format xyz||pdb]  [frame <int frame>] \
          [select [not] \
               [name <string atomname>] \
               [rname <string residue-name>]
               [id  <int atom-id>|<int range atom-id1:atom-id2> ... ]
               [resi <int residue-id>|<int range residue-id1:residue-id2> ... ]
         ]
   <string filename> ]
*/

FORMAT: 'FORMAT' | 'format';
XYZ: 'XYZ' | 'xyz';
PDB: 'PDB' | 'pdb';
FRAME: 'FRAME' | 'frame';
range_spec: ICONST ':' ICONST;
SELECT: 'SELECT' | 'select';
NOT: 'NOT' | 'not';
NAME: 'NAME' | 'name';
RNAME: 'RNAME' | 'rname';
KW_ID: 'id' | 'ID';
RESI: 'RESI' | 'resi';

load_format: FORMAT (XYZ| PDB);
load_directive: LOAD (load_format)? (FRAME ICONST)?
                (SELECT (NOT)? (NAME ID)? RNAME ID
                (KW_ID ICONST | range_spec)*
                (RESI ICONST | range_spec)*
                )?
                ID;

/*
   [ZMATRIX || ZMT || ZMAT
        <string tagn> <list_of_zmatrix_variables>
        ...
        [VARIABLES
             <string symbol> <real value>
             ... ]
        [CONSTANTS
             <string symbol> <real value>
             ... ]
    (END || ZEND)]
*/

ZMAT_TAG: 'tag'ICONST;
ZMATRIX: 'ZMATRIX' | 'ZMT' | 'ZMAT' | 'zmatrix' | 'zmt' | 'zmat';
VARIABLES: 'VARIABLES' | 'variables';
CONSTANTS: 'CONSTANTS' | 'constants';
END: 'END' | 'end';
ZEND: 'ZEND' | 'zend';

zmatrix_directive:
                ZMATRIX (ZMAT_TAG ID+)+
                (VARIABLES (ID FCONST)+)?
                (CONSTANTS (ID FCONST)+)?
                (END | ZEND);
/*
    [ZCOORD
         CVR_SCALING <real value>
         BOND    <integer i> <integer j> \
                 [<real value>] [<string name>] [constant]
         ANGLE   <integer i> <integer j> \
                     [<real value>] [<string name>] [constant]
         TORSION <integer i> <integer j> <integer k> <integer l> \
                 [<real value>] [<string name>] [constant]
     END]
*/

BOND: 'BOND' | 'bond';
ZCOORD: 'ZCOORD' | 'zcoord';
ANGLE: 'ANGLE' | 'angle';
TORSION: 'TORSION' | 'torsion';
CONSTANT: 'CONSTANT' | 'constant';
CVR_SCALING: 'CVR_SCALING' | 'cvr_scaling';

zcoord_directive: ZCOORD CVR_SCALING FCONST
                  BOND ICONST ICONST FCONST? ID? CONSTANT?
                  ANGLE ICONST ICONST FCONST? ID? CONSTANT?
                  TORSION ICONST ICONST ICONST ICONST FCONST? CONSTANT? END;

//[SYSTEM surface  <molecule polymer surface crystal default molecule>
//          lat_a <real lat_a> lat_b <real lat_b> lat_c <real lat_c>
//          alpha <real alpha> beta <real beta> gamma <real gamma>
//     END]

MOLECULE: 'MOLECULE'|'molecule';
POLYMER: 'POLYMER' | 'polymer';
SURFACE: 'surface' | 'SURFACE';
CRYSTAL: 'CRYSTAL' | 'crystal';
SYSTEM: 'SYSTEM' | 'system';
LAT_A: 'LAT_A' | 'lat_a';
LAT_B: 'LAT_B' | 'lat_b';
LAT_C: 'LAT_C' | 'lat_c';
ALPHA: 'ALPHA' | 'alpha';
BETA: 'BETA' | 'beta';
GAMMA: 'GAMMA' | 'gamma';

system_directive: SYSTEM (MOLECULE|POLYMER|SURFACE|CRYSTAL)?
                  LAT_A FCONST LAT_B FCONST LAT_C FCONST
                  ALPHA FCONST BETA FCONST GAMMA FCONST;

geometry_directive: GEOMETRY ID?
                    (UNITS ID?)?
                    (ID FCONST?)?
                    (PRINT ID? | NOPRINT)?
                    (CENTER | NOCENTER)?
                    (BQBQ)? (AUTOSYM (TOL FCONST)? | NOAUTOSYM)?
                    (AUTOZ | NOAUTOZ)? (ADJUST)?
                    (NUCLEUS ID )?
                    symmetry_directive? load_directive?
                    cartesian_coord_directive? zmatrix_directive?
                    zcoord_directive? system_directive?
                    ;


CENTER: 'CENTER' | 'center';
NOCENTER: 'NOCENTER' | 'nocenter';
BQBQ: 'BQBQ' | 'bqbq';
AUTOSYM: 'AUTOSYM' | 'autosym';
TOL: 'TOL' | 'tol';
NOAUTOSYM: 'NOAUTOSYM' | 'noautosym';
AUTOZ: 'AUTOZ' | 'autoz';
NOAUTOZ: 'NOAUTOZ' | 'noautoz';
ADJUST: 'ADJUST' | 'adjust';
//NUC: 'NUC' | 'nuc';
//NUCL: 'NUCL' | 'nucl';
NUCLEUS: 'NUCLEUS' | 'nucleus' | 'NUC' | 'nuc' | 'NUCL' | 'nucl';

IGNORE: 'IGNORE'  | 'ignore';

task_directive: TASK ID (ID)? IGNORE?;

/*
BSSE
 MON <string monomer name> <integer natoms>
 [INPUT [<string input>]]
 [INPUT_WGHOST[<string input>]]
 [CHARGE [<real charge>]]
 [ MULT <integer mult>]
 [OFF]
 [ON]
END
*/

BSSE: 'BSSE' | 'bsse';
MON: 'MON' | 'mon';
INPUT : 'INPUT' | 'input';
INPUT_WGHOST: 'INPUT_WGHOST' | 'input_wghost';
MULT: 'MULT' | 'mult';
ON: 'ON' | 'on';
OFF: 'OFF' | 'off';

bsse_directive: BSSE MON ID ICONST
                INPUT ID INPUT_WGHOST ID
                (CHARGE FCONST)? (MULT ICONST)?
                OFF? ON? END;


Whitespace
    :   [ \t]+
        -> skip
    ;

Newline
    :   (   '\r' '\n'?
        |   '\n'
        )
        -> skip
    ;

BlockComment
    :   '/*' .*? '*/'
        -> skip
    ;

LineComment
    :   '//' ~[\r\n]*
        -> skip
    ;
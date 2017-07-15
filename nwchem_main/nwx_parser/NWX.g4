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


// Identifier
//ID
//    :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
//    ;

//Can be file name, dir path or any other name
ID
        :('a'..'z'|'A'..'Z'|'0'..'9'|'_'|'/'|'.'|'-')+ ;

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


/*   memory [[total] <total_size>] [stack <stack_size>]
            [heap <heap_size>] [global <global_size>]
            [<units>] [verify|noverify] [hardfail|nohardfail]
*/

//memory_spec: MEMORY


// translation-unit
nwchem_input: directive_list EOF ;

directive_list: (
                start_directive
                );

/*
(RESTART || START) [<string file_prefix default input_file_prefix>] \
                   [rtdb <string rtdb_file_name default file_prefix.db>]
*/
start_directive:  (START | RESTART) ID? (RTDB ID?)? ;

// [SYMMETRY [group] <string group_name>|<integer group number> [setting <integer setting>] [print] \
//          [tol <real tol default 1d-2>]]

symmetry_directive: ;


/* [ LOAD [format xyz||pdb]  [frame <int frame>] \
          [select [not] \
               [name <string atomname>] \
               [rname <string residue-name>]
               [id  <int atom-id>|<int range atom-id1:atom-id2> ... ]
               [resi <int residue-id>|<int range residue-id1:residue-id2> ... ]
         ]
   <string filename> ]
*/
load_directive: ;



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
zmatrix_directive:;

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
zcoord_directive: ;


//[SYSTEM surface  <molecule polymer surface crystal default molecule>
//          lat_a <real lat_a> lat_b <real lat_b> lat_c <real lat_c>
//          alpha <real alpha> beta <real beta> gamma <real gamma>
//     END]

system_directive:;

geometry_directive: GEOMETRY ID?
                    (UNITS ID?)?
                    (ID FCONST?)?
                    (PRINT ID? | NOPRINT)?
                    (CENTER | NOCENTER)?
                    (BQBQ)? (AUTOSYM (TOL FCONST)? | NOAUTOSYM)?
                    (AUTOZ | NOAUTOZ)? (ADJUST)?
                    ( (NUC | NUCL | NUCLEUS) ID )?
                    symmetry_directive? load_directive? /* string tag */
                    zmatrix_directive? zcoord_directive?
                    system_directive?

                    ;


CENTER: 'CENTER' | 'center';
NOCENTER: 'NOCENTER' | 'nocenter';

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
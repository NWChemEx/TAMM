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


// Identifier
//ID
//    :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
//    ;

FILE_NAME
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
start_directive:  (START | RESTART) FILE_NAME? (RTDB FILE_NAME?)? ;

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
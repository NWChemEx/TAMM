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
ENERGY: 'ENERGY' | 'energy';

GEOMETRY: 'GEOMETRY' | 'geometry';
SYMMETRY: 'SYMMETRY' | 'symmetry';
LOAD: 'LOAD' | 'load';
RTDB: 'RTDB' | 'rtdb';
CHARGE: 'CHARGE' | 'charge';
UNITS: 'UNITS' | 'units';
END: 'END' | 'end';
COLON: ':';
STAR: '*';

// Memory Directive
TOTAL: 'TOTAL' | 'total';
STACK: 'STACK' | 'stack';
HEAP: 'HEAP' | 'heap';
GLOBAL: 'GLOBAL' | 'global';
VERIFY: 'VERIFY' | 'verify';
NOVERIFY: 'NOVERIFY' | 'noverify';
HARDFAIL: 'HARDFAIL' | 'hardfail';
NOHARDFAIL: 'NOHARDFAIL' | 'nohardfail';

// SCF Directive
SCF: 'SCF' | 'scf';
SINGLET: 'SINGLET' | 'singlet';
DOUBLET: 'DOUBLET' | 'doublet';
TRIPLET: 'TRIPLET' | 'triplet';
QUARTET: 'QUARTET' | 'quartet';
QUINTET: 'QUINTET' | 'quintet';
SEXTET: 'SEXTET' | 'sextet';
SEPTET: 'SEPTET' | 'septet';
OCTET: 'OCTET' | 'octet';
NOPEN: 'NOPEN' | 'nopen';
RHF: 'RHF' | 'rhf';
ROHF: 'ROHF' | 'rohf';
UHF: 'UHF' | 'uhf';

// CCSD Directive
CCSD: 'CCSD' | 'ccsd';
MAXITER: 'MAXITER' | 'maxiter';
THRESH: 'THRESH' | 'thresh';
DIISBAS: 'DIISBAS' | 'diisbas';
NODISK: 'NODISK' | 'nodisk';
IPRT: 'IPRT' | 'iprt';
FREEZE: 'FREEZE' | 'freeze';
ATOMIC: 'ATOMIC' | 'atomic';
VIRTUAL: 'VIRTUAL' | 'virtual';
CORE: 'CORE' | 'core';


// Basis Set
 SPHERICAL: 'SPHERICAL' | 'spherical';
 CARTESIAN: 'CARTESIAN' | 'cartesian';
 REL: 'REL' | 'rel';
 FILE: 'FILE' | 'file';
 BASIS: 'BASIS' | 'basis';
 LIBRARY: 'LIBRARY' | 'library';
 EXCEPT: 'EXCEPT' | 'except';

// TCE Directives
LCCD: 'LCCD' | 'lccd';
CCD: 'CCD' | 'ccd';
CC2: 'CC2' | 'cc2';
LRCCSD: 'LR-CCSD' | 'lr-ccsd';
LCCSD: 'LCCSD' | 'lccsd';
CCSDT: 'CCSDT' | 'ccsdt';
CCSDTA: 'CCSDTA' | 'ccsdta';
CCSDTQ: 'CCSDTQ' | 'ccsdtq';
CCSDPT: 'CCSD(T)' | 'ccsd(t)';
CCSDBT: 'CCSD[T]' | 'ccsd[t]';

CCSD2: 'CCSD2' | 'ccsd2';
CCSD2T: 'CCSD(2)_T' | 'ccsd(2)_t';

CCSDT2Q: 'CCSDT(2)_Q' | 'ccsdt(2)_q';

CRCCSDBT: 'CR-CCSD[T]' | 'cr-ccsd[t]';
CRCCSDPT: 'CR-CCSD(T)' | 'cr-ccsd(t)';

LRCCSDPT: 'LR-CCSD(T)' | 'lr-ccsd(t)';
LRCCSDPTQ1: 'LR-CCSD(TQ)-1' | 'lr-ccsd(tq)-1';

CREOMSDPT: 'CREOMSD(T)' | 'creomsd(t)';

QCISD: 'QCISD' | 'qcisd';
CISD: 'CISD' | 'cisd';
CISDT: 'CISDT' | 'cisdt';
CISDTQ: 'CISDTQ' | 'cisdtq';
MBPT2: 'MBPT2' | 'mbpt2';

MBPT3: 'MBPT3' | 'mbpt3';
MBPT4: 'MBPT4' | 'mbpt4';
MP2: 'MP2' | 'mp2';
MP3: 'MP3' | 'mp3';
MP4: 'MP4' | 'mp4';

// High-level

TCE: 'TCE' | 'tce';
HF: 'HF' | 'hf';
DFT: 'DFT' | 'dft';


TWO_EORB: '2EORB' | '2eorb';
TOL2E: 'TOL2E' | 'tol2e';
TWO_EMET: 'TWO_EMET' | 'two_emet';
IO: 'IO' | 'io';
DIIS: 'DIIS' | 'diis';
LSHIFT: 'LSHIFT' | 'lshift';
NROOTS: 'NROOTS' | 'nroots';
TARGET: 'TARGET' | 'target';
TARGETSYM: 'TARGETSYM' | 'targetsym';
T3A_LVL: 'T3A_LVL' | 't3a_lvl';
ACTIVE_OA: 'ACTIVE_OA' | 'active_oa';
ACTIVE_OB: 'ACTIVE_OB' | 'active_ob';
ACTIVE_VA: 'ACTIVE_VA' | 'active_va';
ACTIVE_VB: 'ACTIVE_VB' | 'active_vb';
DIPOLE: 'DIPOLE' | 'dipole';
TILESIZE: 'TILESIZE' | 'tilesize';
FOCK: 'FOCK' | 'fock';
NOFOCK: 'NOFOCK' | 'nofock';
FRAGMENT: 'FRAGMENT' | 'fragment';

// Identifier can be file name, dir path or any other name
ID
        :[a-zA-Z] [a-zA-Z0-9_-]*;
        //('a'..'z'|'A'..'Z'|'0'..'9'|'_')+; //('a'..'z'|'A'..'Z'|'0'..'9'|'_'|'/'|'.'|'-')*;
         //| '"' ID '"';

//DIR_FILE_PATH
//    : ('a'..'z'|'A'..'Z'|'0'..'9'|'_'|'/'|'.'|'-')+
//    ;


// Integer Constant
ICONST
    :   [0-9]+
    ;

FRAC
    :   [1-9]+ '/' [1-9]+
    ;

// Foalting Point Constant
FCONST
    :   '-'?
    ( [0-9]+ '.' [0-9]* EXPONENT?
    |   '.'  [0-9]+ EXPONENT?
    |    [0-9]+ EXPONENT
    )
    ;

fragment EXPONENT
    :   ('e'|'E'|'d'|'D') ('+'|'-')?  [0-9]+
    ;

//fragment NEWLINE   : '\r' '\n' | '\n' | '\r';
//
//: NEWLINE*;

//TODO: http://www.nwchem-sw.org/index.php/Release66:NWChem_Documentation

// translation-unit
nwchem_input: directive_list EOF;

directive_list:
                (start_directive
                | permanent_dir_directive
                | scratch_dir_directive
                | ECHO

                | basis_directive
                | memory_directive
                | title_directive
                | print_directive
                | set_directive
                | unset_directive
                | STOP
                | geometry_directive
                | scf_directive
                | tce_directive
                | task_directive
                | ecce_print_directive
                )*;

 /*
 BASIS [<string name default "ao basis">] \
       [(spherical || cartesian) default cartesian] \
       [(print || noprint) default print]
       [rel]
    <string tag> library [<string tag_in_lib>] \
                 <string standard_set> [file <filename>] \
                 [except <string tag list>] [rel]
       ...
    <string tag> <string shell_type> [rel]
       <real exponent> <real list_of_coefficients>
       ...
 END
 */





 basis_directive: BASIS ID? (SPHERICAL | CARTESIAN)? (PRINT | NOPRINT)? REL?
                  (ID|STAR) LIBRARY ID?
                  ID (FILE ID)?
                  (EXCEPT ID+)? REL?

                  (ID ID REL? FCONST FCONST)*
                  END;

/* Effective Core Potentials
 ECP [<string name default "ecp basis">] \
       [print || noprint default print]
    <string tag> library [<string tag_in_lib>] \
                 <string standard_set> [file <filename>] \
                 [except <string tag list>]
    <string tag> [nelec] <integer number_of_electrons_replaced>
       ...
    <string tag> <string shell_type>
    <real r-exponent> <real Gaussian-exponent> <real list_of_coefficients>
       ...
 END
 */

ECP: 'ECP' | 'ecp';
NELEC: 'NELEC' | 'nelec';

 ecp_directive: ECP ID?
                ID LIBRARY ID?
                ID (FILE ID)?
                (EXCEPT ID+)?
                (ID NELEC? ICONST)+
                ( ID ID (FCONST FCONST FCONST+)+ )+
                END;

/* Spin-orbit ECP
 SO [<string name default "so basis">] \
       [print || noprint default print]
    <string tag> library [<string tag_in_lib>] \
                 <string standard_set> [file <filename>]
                 [except <string tag list>]
       ...
    <string tag> <string shell_type>
    <real r-exponent> <real Gaussian-exponent> <real list_of_coefficients>
       ...
 END
*/
 so_ecp_directive: ('so' | 'SO') ID? (PRINT | NOPRINT)?
                   ID LIBRARY ID?
                   ID (FILE ID)?
                   (EXCEPT ID+)?
                   ( ID ID (FCONST FCONST FCONST+)+ )+
                   END;

/*
  RELATIVISTIC
  [DOUGLAS-KROLL [<string (ON||OFF) default ON> \
                <string (FPP||DKH||DKFULL||DK3||DK3FULL) default DKH>]  ||
   ZORA [ (ON || OFF) default ON ] ||
   DYALL-MOD-DIRAC [ (ON || OFF) default ON ]
                 [ (NESC1E || NESC2E) default NESC1E ] ]
  [CLIGHT <real clight default 137.0359895>]
 END
 */

DYALL_MOD_DIRAC: 'DYALL-MOD-DIRAC' | 'dyall-mod-dirac';
RELATIVISTIC: 'RELATIVISTIC' | 'relativistic';
FPP: 'FPP' | 'kpp';
DKH: 'DKH' | 'dkh';
DKFULL: 'DKFULL' | 'dkfull';
DK3: 'DK3' | 'dk3';
DK3FULL: 'DK3FULL' | 'dk3full';
ZORA: 'ZORA' | 'zora';

zora_directive: ZORA (ON | OFF)?;
NESC1E: 'NESC1E' | 'nesc1e';
NESC2E: 'NESC2E' | 'nesc2e';
CLIGHT: 'CLIGHT' | 'clight';

DOUGLAS_KROLL: 'DOUGLAS-KROLL' | 'douglas-kroll';

clight_directive: CLIGHT FCONST;
dyall_mod_dirac_directive: DYALL_MOD_DIRAC (ON | OFF)? (NESC1E | NESC2E)?;

relativistic_electron_approx_directive:
            RELATIVISTIC
            (
               ID (ON | OFF)
               ID (FPP | DKH | DKFULL | DK3 | DK3FULL)
            )?
            |
             zora_directive?
            |
            dyall_mod_dirac_directive
            clight_directive?
            END;

/*
 DOUGLAS-KROLL [<string (ON||OFF) default ON> \
                <string (FPP||DKH||DKFULL|DK3|DK3FULL) default DKH>]
*/

douglas_kroll_directive: DOUGLAS_KROLL
                         (
                          ID (ON | OFF)
                          ID (FPP | DKH | DKFULL | DK3 | DK3FULL)
                         )?;

/*   memory [[total] <total_size>] [stack <stack_size>]
            [heap <heap_size>] [global <global_size>]
            [<units>] [verify|noverify] [hardfail|nohardfail]
*/



memory_directive: MEMORY (TOTAL? ICONST)? (STACK ICONST ID?)?
                  (HEAP ICONST ID?)? (GLOBAL ICONST ID?)?
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

charge_directive: (CHARGE FCONST);

/*
(RESTART || START) [<string file_prefix default input_file_prefix>] \
                   [rtdb <string rtdb_file_name default file_prefix.db>]
*/
start_directive:  (START | RESTART) ID? (RTDB ID?)?  ;

ecce_print_directive: ECCE_PRINT ID ;

permanent_dir_directive: PERMANENT_DIR ( ( (ID | ICONST) COLON)? ID)+ ;

scratch_dir_directive: SCRATCH_DIR ( ( (ID | ICONST) COLON)? ID)+ ;

SETTING: 'SETTING' | 'setting';
GROUP: 'GROUP' | 'group';

// [SYMMETRY [group] <string group_name>|<integer group number> [setting <integer setting>] [print] \
//          [tol <real tol default 1d-2>]]

symmetry_directive: SYMMETRY  GROUP? (ID | ICONST) (SETTING ICONST)? PRINT? (TOL FCONST)? ;

SYM: 'SYM' | 'sym';
ADAPT : 'ADAPT' | 'adapt';
sym_switch: SYM (ON|OFF)?; //FIXME: can it be symmetry keyword?
adapt_switch: ADAPT (ON | OFF)?;



// TOL2E <real tol2e default min(10e-7 , 0.01*thresh)>
tolerance_directive: TOL2E FCONST?;

/*
   [<string tag> <real x y z> [vx vy vz] [charge <real charge>] \
          [mass <real mass>] \
          [(nuc || nucl || nucleus) <string nucmodel>]
   ... ]
*/

MASS: 'MASS' | 'mass';

cartesian_coord_directive: (ID FCONST FCONST FCONST)+;


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
                    (UNITS ID)?  //check if ID can be optional
                    (ID FCONST?)?
                    (PRINT ID? | NOPRINT)?
                    (CENTER | NOCENTER)?
                    (BQBQ)? (AUTOSYM (TOL FCONST)? | NOAUTOSYM)?
                    (AUTOZ | NOAUTOZ)? (ADJUST)?
                    (NUCLEUS ID )?
                    symmetry_directive? load_directive?
                    cartesian_coord_directive? zmatrix_directive?
                    zcoord_directive? system_directive? END
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
GRADIENT: 'gradient' | 'GRADIENT';
OPTIMIZE: 'OPTIMIZE' | 'optimize';
SADDLE: 'saddle' | 'SADDLE';
HESSIAN: 'hessian' | 'HESSIAN';
FREQ: 'freq' | 'FREQ';
FREQUENCIES: 'frequencies' | 'FREQUENCIES';
VSCF: 'vscf' | 'VSCF';
PROPERTY: 'property' | 'PROPERTY';
DYNAMICS: 'dynamics' | 'DYNAMICS';
THERMODYNAMICS: 'thermodynamics' | 'THERMODYNAMICS';

task_directive: TASK
    ( SCF | DFT | SODFT | MP2 | DIRECT_MP2 | RIMP2 | CCSD | CCSD_T
     | MSSCF | SELCI | MD | PSPW | BAND | TCE)
    (ENERGY | GRADIENT | OPTIMIZE | SADDLE | HESSIAN | FREQ | FREQUENCIES |
     VSCF | PROPERTY | DYNAMICS | THERMODYNAMICS)? IGNORE?;

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
                charge_directive? (MULT ICONST)?
                OFF? ON? END;

//http://www.nwchem-sw.org/index.php/Release66:Hartree-Fock_Theory_for_Molecules


wavefn_type:                  SINGLET?
                              DOUBLET?
                              TRIPLET?
                              QUARTET?
                              QUINTET?
                              SEXTET?
                              SEPTET?
                              OCTET?
                              (NOPEN ICONST)?
                              (RHF | ROHF | UHF);

scf_directive: SCF
                (
                  (SYM (ON|OFF))
                | (TOL2E FCONST)
                | (ADAPT (ON|OFF))
                | (THRESH FCONST)
                | wavefn_type
                )*
               END;

/*
 VECTORS [[input] (<string input_movecs default atomic>) || \
                  (project <string basisname> <string filename>) || \
                  (fragment <string file1> [<string file2> ...])] \
         [swap [alpha||beta] <integer vec1 vec2> ...] \
         [reorder <integer atom1 atom2> ...] \
         [output <string output_filename default input_movecs>] \
         [lock]
         [rotate  <string input_geometry> <string input_movecs>]
*/

//mo_vectors_directive:;


/*Coupled Cluster Directives
----------------------------
 CCSD
   [MAXITER <integer maxiter default 20>]
   [THRESH  <real thresh default 10e-6>]
   [TOL2E <real tol2e default min(10e-12 , 0.01*$thresh$)>]
   [DIISBAS  <integer diisbas default 5>]
   [FREEZE [[core] (atomic || <integer nfzc default 0>)] \
           [virtual <integer nfzv default 0>]]
   [NODISK]
   [IPRT  <integer IPRT default 0>]
   [PRINT ...]
   [NOPRINT ...]
 END
*/




maxiter_directive: MAXITER ICONST?;
threshold_directive: THRESH FCONST?;
freeze_directive: FREEZE
                  (CORE? (ATOMIC | ICONST))?
                  (VIRTUAL ICONST)?;


ccsd_directive: CCSD
                maxiter_directive?
                threshold_directive?
                tolerance_directive?
                (DIISBAS ICONST?)?
                freeze_directive?
                NODISK?
                (IPRT ICONST?)?
                print_directive? //FIXME: Check exact syntax
                END;

/* End Coupled Cluster Directives */


/* Tensor Contraction Engine (TCE) directives
---------------------------------------------
 TCE
   [(DFT||HF||SCF) default HF=SCF]
   [FREEZE [[core] (atomic || <integer nfzc default 0>)] \
            [virtual <integer nfzv default 0>]]
   [(LCCD||CCD||CCSD||CC2||LR-CCSD||LCCSD||CCSDT||CCSDTA||CCSDTQ|| \
     CCSD(T)||CCSD[T]||CCSD(2)_T||CCSD(2)||CCSDT(2)_Q|| \
     CR-CCSD[T]||CR-CCSD(T)|| \
     LR-CCSD(T)||LR-CCSD(TQ)-1||CREOMSD(T)|| \
     QCISD||CISD||CISDT||CISDTQ|| \
     MBPT2||MBPT3||MBPT4||MP2||MP3||MP4) default CCSD]
   [THRESH <double thresh default 1e-6>]
   [MAXITER <integer maxiter default 100>]
   [PRINT (none||low||medium||high||debug)
     <string list_of_names ...>]
   [IO (fortran||eaf||ga||sf||replicated||dra||ga_eaf) default ga]
   [DIIS <integer diis default 5>]
   [LSHIFT <double lshift default is 0.0d0>]
   [NROOTS <integer nroots default 0>]
   [TARGET <integer target default 1>]
   [TARGETSYM <character targetsym default 'none'>]
   [SYMMETRY]
   [2EORB]
   [2EMET <integer fast2e default 1>]
   [T3A_LVL]
   [ACTIVE_OA]
   [ACTIVE_OB]
   [ACTIVE_VA]
   [ACTIVE_VB]
   [DIPOLE]
   [TILESIZE <no default (automatically adjusted)>]
   [(NO)FOCK <logical recompf default .true.>]
   [FRAGMENT <default -1 (off)>]
 END
*/



logical_directive: '.true.' | '.false' | '.TRUE.' | '.FALSE';

tce_theories:
 LCCD | CCD | CCSD | CC2 | LRCCSD | LCCSD | CCSDT | CCSDTA |
     CCSDTQ | CCSDPT | CCSDBT | CCSD2T | CCSD2 | CCSDT2Q |
     CRCCSDBT | CRCCSDPT |
     LRCCSDPT | LRCCSDPTQ1 | CREOMSDPT |
     QCISD | CISD | CISDT | CISDTQ |
     MBPT2 | MBPT3 | MBPT4 | MP2 | MP3 | MP4;



tce_directive: TCE

   (
   (DFT | HF | SCF)
   | freeze_directive
   | tce_theories
   | threshold_directive
   | maxiter_directive
   | print_directive
   | (IO ('fortran' | 'eaf' | 'ga' | 'sf' | 'replicated' | 'dra' | 'ga_eaf') )
   | (DIIS ICONST?)
   | (LSHIFT FCONST?)
   | (NROOTS ICONST?)
   | (TARGET ICONST?)
   | (TARGETSYM ID?) //FIXME: check syntax
   | (SYMMETRY)
   | (TWO_EORB)
   | (TWO_EMET ICONST?)
   | (T3A_LVL)
   | (ACTIVE_OA)
   | (ACTIVE_OB)
   | (ACTIVE_VA)
   | (ACTIVE_VB)
   | (DIPOLE)
   | (TILESIZE ICONST) //FIXME: CHECK
   | ( (FOCK | NOFOCK) logical_directive)
   | (FRAGMENT ICONST)
   )*
   END;


WS : [ \t\r\n]+ -> skip ; // skip spaces, tabs, newlines


BlockComment
    :   '/*' .*? '*/'
        -> skip
    ;

LineComment
    :   '#' ~[\r\n]*
        -> skip
    ;
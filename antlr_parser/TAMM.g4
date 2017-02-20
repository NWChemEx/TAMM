//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//------------------------------------------------------------------------------

grammar TAMM;


// TOKENS 
    
// Reserved Keywords
RANGE   :   'range';
INDEX  :   'index';
ARRAY  :   'array';
SCALAR :   'scalar';

// Operators
PLUS    :   '+';
MINUS   :   '-';
TIMES   :   '*';

// Assignment operators
EQUALS   :   '=';
TIMESEQUAL : '*=';
PLUSEQUAL : '+=';
MINUSEQUAL : '-=';

// Delimeters
LPAREN  :   '(';
RPAREN  :   ')';
LBRACE  :   '{';
RBRACE  :   '}';
LBRACKET  :   '[';
RBRACKET  :   ']';
COMMA   :   ',';
COLON   :   ':';
SEMI:  ';';

// Identifier   
ID
    :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')*
    ;

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
translation_unit : compound_element_list EOF ;

compound_element_list: (compound_element)* ;

// compound-element
compound_element : identifier LBRACE element_list RBRACE ;

element_list: (element)* ;        

// element
element : 
         declaration 
         |
         statement ;


// declaration
declaration : range_declaration 
              |       
              index_declaration 
              |
              scalar_declaration
              |
              array_declaration ;


scalar_declaration: SCALAR ID (COMMA ID)* SEMI;

// id-list
id_list_opt : 
             |
             id_list ;


id_list : identifier (COMMA identifier)*;
              
num_list : numerical_constant (COMMA numerical_constant)*;       


// identifier
identifier : ID ;
    

// numerical-constant
numerical_constant : ICONST 
                     |
                     FCONST
                     |
                     FRAC;
    
    
    
// range-declaration
range_declaration : RANGE id_list EQUALS numerical_constant SEMI ;


// index-declaration
index_declaration : INDEX id_list EQUALS identifier SEMI ;


// array-declaration
array_declaration : ARRAY array_structure_list (COLON identifier)? SEMI ;


array_structure : ID LBRACKET id_list_opt RBRACKET LBRACKET id_list_opt RBRACKET; //(permut_symmetry)?;

array_structure_list : array_structure (COMMA array_structure)* ;



// statement
statement : assignment_statement ;


// assignment-statement
assignment_statement : (identifier COLON)? array_reference assignment_operator expression SEMI ;


// assignment_operator
assignment_operator : EQUALS
                           | TIMESEQUAL
                           | PLUSEQUAL
                           | MINUSEQUAL ;
                           

    
// primary-expression    
unary_expression :   numerical_constant 
                     |
                     array_reference 
                     |
                     LPAREN expression RPAREN ;


// array-reference
array_reference : ID (LBRACKET id_list RBRACKET)? ;


// expression                           
plusORminus : PLUS | MINUS ;

// additive-expression
expression : (plusORminus)? multiplicative_expression (plusORminus multiplicative_expression)* ;


// multiplicative-expression
multiplicative_expression : unary_expression (TIMES unary_expression)* ;
                            

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


// // permutational-symmetry
// permut_symmetry : COLON (symmetry_group)+ ;
                          
                            
// symmetry_group : LPAREN num_list RPAREN ;

// expansion_declaration : EXPAND id_list SEMI ;

// // volatile-declaration
// volatile_declaration : VOLATILE id_list SEMI ;

// // iteration-declaration
// iteration_declaration : ITERATION EQUALS numerical_constant SEMI ;
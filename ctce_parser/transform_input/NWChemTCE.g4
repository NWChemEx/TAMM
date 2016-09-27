grammar NWChemTCE;

options {
    language = Java;
}

// TOKENS

// Reserved Keywords
SUM     : 'Sum' ;

// Operators
PLUS    :   '+';
MINUS   :   '-';
TIMES   :   '*';
EQUALS   :   '=';

// Delimeters
LPAREN  :   '(';
RPAREN  :   ')';
LBRACKET  :   '[';
RBRACKET  :   ']';
SYMOP : '=>' ;
SEMI : ';';

// Identifiers

/*// Array Type  P, v, t, other
ArrType : 'P' | 'p' | 'a'..'o' | 'q'..'z' | 'A'..'O' | 'Q'..'Z' ;

// Electron Type (particle, hole, general)
Etype : ('h' | 'p' | 'a'..'g' | 'i'..'o' | 'q'..'z') ICONST TIMES? ;*/

//Etype : ('a'..'z') ICONST TIMES? ;

// Identifier
ID
    :   ('a'..'z'|'A'..'Z'|'_') ('a'..'z'|'A'..'Z'|'0'..'9'|'_')* TIMES?
    ;

// Integer Constant
ICONST
    :   '0'..'9'+
    ;

// Foalting Point Constant
FCONST
    :   ('0'..'9')+ '.' ('0'..'9')* EXPONENT?
    |   '.' ('0'..'9')+ EXPONENT?
    |   ('0'..'9')+ EXPONENT
    ;

FRAC
    :   ('1'..'9')+ '/' ('1'..'9')+
    ;

fragment EXPONENT
    :   ('e'|'E') ('+'|'-')? ('0'..'9')+
    ;


assignment_operator :
           EQUALS | PLUS EQUALS | MINUS EQUALS | TIMES EQUALS
          ;

// numerical-constant
numerical_constant : ICONST
                     |
                     FCONST
                      |
                      FRAC;

// Plus or Minus
plusORminus : PLUS | MINUS ;


// translation-unit
translation_unit : compound_element_list_opt  ;


// compound-element-list
compound_element_list_opt : EOF
                           |
                           compound_element_list ;


compound_element_list : (statement)+ ;

//factors :  factors_opt+  ;

statement :  array_reference assignment_operator  ptype;

perm : 'P' LPAREN numerical_constant RPAREN;

ptype : ( TIMES (perm | sumExp | array_reference ) | (plusORminus? numerical_constant)+ )+ SEMI;

//arrDims : LPAREN ( SYMOP? ID ) * RPAREN  ;

sumExp : (SUM LPAREN identifier* RPAREN )  ;

// Removed left-recursion from id_list
id_list : (identifier)* ;

// identifier
identifier : ID;

// array-reference
array_reference : ID
                  |
                  ID LPAREN id_list RPAREN identifier;

// Comments
COMMENT
    :  ( '#' ~( '\r' | '\n' )*
    | '/*' (.)*? '*/') -> skip
    ;


// Ignored characters
WS : [ \r\t\n]+ -> skip ;


/*
statement : assignment_statement ;


// assignment-statement
assignment_statement : array_reference assignment_operator expression SEMI ;


// unary-expression
unary_expression : primary_expression
                   |
                   PLUS unary_expression
                   |
                   MINUS unary_expression ;


// primary-expression
primary_expression : numerical_constant
                     |
                     sumExp
                     |
                     array_reference
                     |
                     LPAREN expression RPAREN ;


// expression
expression : additive_expression ;


// Removed left-recursion from additive expression
additive_expression :  multiplicative_expression additive_expression_prime ;

additive_expression_prime :
                           |
                           plusORminus multiplicative_expression additive_expression_prime ;


// Removed left-recursion from multiplicative expression
multiplicative_expression : unary_expression multiplicative_expression_prime ;

multiplicative_expression_prime :
                                 |
                                 TIMES unary_expression multiplicative_expression_prime ;*/

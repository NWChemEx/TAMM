grammar OpMin;

options {
    language = Java;
}

/*@parser::includes {
    #include <iostream>
    #include <list>
    #include <sstream>
    #include <string>
    using namespace std;
}*/


// TOKENS 
    
// Reserved Keywords
RANGE   :   'range';
INDEX  :   'index';
ARRAY  :   'array';
EXPAND :   'expand';
VOLATILE : 'volatile';
ITERATION : 'iteration';

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
translation_unit : compound_element_list_opt EOF ;


// compound-element-list
compound_element_list_opt : 
                           |
                           compound_element_list ;


// Remove left-recursion from compound_element_list
//compound_element_list : compound_element 
//                        |
//                        compound_element_list compound_element ;


// Removed left-recursion from compound_element_list
compound_element_list : compound_element compound_element_list_prime ;


compound_element_list_prime : 
                             | 
                             compound_element compound_element_list_prime ;
                             
                             


// compound-element
compound_element : identifier? LBRACE element_list_opt RBRACE ;


// element-list
element_list_opt : 
                 |
                 element_list ;

// Remove left-recursion from element list
//element_list : element 
//               |
//               element_list element ;


// Removed left-recursion from element list              
element_list : element element_list_prime ;
               
element_list_prime :
                    |
                    element element_list_prime ;               

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
              array_declaration 
              |
              expansion_declaration 
              |
              volatile_declaration 
              |
              iteration_declaration ;


// id-list
id_list_opt : 
             |
             id_list ;
    
    
// Remove left-recursion from id_list    
//id_list : identifier 
//         |
//         id_list COMMA identifier ;


// Removed left-recursion from id_list
id_list : identifier id_list_prime ;
         
id_list_prime: 
              |
              COMMA identifier id_list_prime ;
              


// num-list
//num_list : numerical_constant 
//           |
//           num_list COMMA numerical_constant ;    
           
           
num_list : numerical_constant num_list_prime ;
           
num_list_prime :
                | 
                COMMA numerical_constant num_list_prime ;       

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


// array-structure-list

// Remove left-recursion from array_structure_list
//array_structure_list : array_structure 
//                       |
//                       array_structure_list COMMA array_structure ;


// Removed left-recursion from array_structure_list
array_structure_list : array_structure array_structure_list_prime ;

array_structure_list_prime :
                            |
                            COMMA array_structure array_structure_list_prime;

// array-structure
// Old - array_structure : ID LPAREN LBRACKET id_list_opt RBRACKET LBRACKET id_list_opt RBRACKET permut_symmetry_opt RPAREN ;

astruct : LBRACKET id_list_opt RBRACKET LBRACKET id_list_opt RBRACKET permut_symmetry_opt;

array_structure : ID
                 (LPAREN  astruct RPAREN
                 |
                 astruct);

// permutational-symmetry
permut_symmetry_opt : 
                     |
                     COLON symmetry_group_list ;


// Remove left-recursion from symmetry_group_list
//symmetry_group_list : symmetry_group 
//                      |
//                      symmetry_group_list symmetry_group ;


// Removed left-recursion from symmetry_group_list
symmetry_group_list : symmetry_group symmetry_group_list_prime ;

symmetry_group_list_prime :
                           |
                           symmetry_group symmetry_group_list_prime ;
                           
                            
symmetry_group : LPAREN num_list RPAREN ;


expansion_declaration : EXPAND id_list SEMI ;



// volatile-declaration
volatile_declaration : VOLATILE id_list SEMI ;


// iteration-declaration
iteration_declaration : ITERATION EQUALS numerical_constant SEMI ;


// statement
statement : assignment_statement ;


// assignment-statement
assignment_statement : (identifier COLON)? expression assignment_operator expression SEMI ;


// assignment_operator
assignment_operator : EQUALS
                           | TIMESEQUAL
                           | PLUSEQUAL
                           | MINUSEQUAL ;
                           

// unary-expression
unary_expression : primary_expression 
                   |
                   PLUS unary_expression 
                   |
                   MINUS unary_expression ;
    
    
    
// primary-expression    
primary_expression : numerical_constant 
                     |
                     array_reference 
                     |
                     LPAREN expression RPAREN ;


// array-reference
array_reference : ID 
                  |
                  ID LBRACKET id_list RBRACKET ;


// expression                           
expression : additive_expression ;


plusORminus : PLUS | MINUS ;

// additive-expression

// Remove left-recursion from additive expression
//additive_expression : multiplicative_expression 
//                      |
//                      additive_expression plusORminus multiplicative_expression 


// Removed left-recursion from additive expression
additive_expression :  multiplicative_expression additive_expression_prime ;

additive_expression_prime : 
                           |
                           plusORminus multiplicative_expression additive_expression_prime ;


// multiplicative-expression

// Remove left-recursion from multiplicative expression
//multiplicative_expression : unary_expression ;
//                            |
//                            multiplicative_expression TIMES unary_expression ;


// Removed left-recursion from multiplicative expression
multiplicative_expression : unary_expression multiplicative_expression_prime ;
                            
multiplicative_expression_prime :
                                 |
                                 TIMES unary_expression multiplicative_expression_prime ;

// Comments
COMMENT
    :  ( '#' ~( '\r' | '\n' )*
    | '/*' (.)*? '*/') -> skip
    ;


// Ignored characters
WS : [ \r\t\n]+ -> skip ;

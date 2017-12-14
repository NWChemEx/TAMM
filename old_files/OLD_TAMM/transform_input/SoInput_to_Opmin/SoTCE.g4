grammar SoTCE;

options {
    language = Python;
}

// TOKENS 
    
// Reserved Keywords
SUM     : 'Sum' ;

// Operators
PLUS    :   '+';
MINUS   :   '-';
TIMES   :   '*';

// Delimeters
LPAREN  :   '(';
RPAREN  :   ')';
LBRACKET  :   '[';
RBRACKET  :   ']';
SYMOP : '=>' ;

// Identifiers

// Array Type  P, v, t, other
ArrType : 'P' | 'p' | 'a'..'o' | 'q'..'z' | 'A'..'O' | 'Q'..'Z' ;

// Electron Type (particle, hole, general)
Etype : ('h' | 'p' | 'a'..'g' | 'i'..'o' | 'q'..'z') ICONST TIMES? ;


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

fragment EXPONENT
    :   ('e'|'E') ('+'|'-')? ('0'..'9')+
    ;
    

// numerical-constant
numerical_constant : ICONST 
                     |
                     FCONST ;

// Plus or Minus    
plusORminus : op=(PLUS | MINUS) ;
    

// translation-unit
translation_unit : compound_element_list_opt  ;


// compound-element-list
compound_element_list_opt : EOF
                           |
                           compound_element_list ;


compound_element_list : (factors)+ ;


factors :  LBRACKET factors_opt+  ;
           
           
factors_opt :  ptype | RBRACKET  ;
           

ptype :  TIMES (sumExp | ArrType arrDims ) | (plusORminus numerical_constant)+;


arrDims : LPAREN ( SYMOP? Etype ) * RPAREN  ;


sumExp : (SUM LPAREN Etype* RPAREN )  ;
 

// Comments
COMMENT
    :  ( '#' ~( '\r' | '\n' )*
    | '/*' (.)*? '*/') -> skip
    ;


// Ignored characters
WS : [ \r\t\n]+ -> skip ;

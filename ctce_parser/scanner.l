%{
#include "parser.h"
#include "error.h"

char *copy(char *s, int len) {
  char *d = (char *)calloc(sizeof(char),len+1);
  strcpy(d,s);
  return d;
}

//suppress ‘input’ defined but not used warning.
#define YY_NO_INPUT

int line = 1, col = 1;
//FIXME: Column nos reported are inaccurate
%}


/*suppress ‘yyunput’ defined but not used warning*/
%option nounput

%option yylineno
%option reentrant noyywrap
%option extra-type="char *"
%option outfile="scanner.c" header-file="scanner.h"

%%

 /* Ignore single-line comments and whitespace. */
"//".*                    { ; }
[ \t]+                  { col+=yyleng;}
\n                      { col = 1; line++; }

range                        { tce_lineno = line; tce_tokPos = col; col += (yyleng); yyextra = copy(yytext,yyleng); return RANGE; }
index                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return INDEX; }
array                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return ARRAY; }
expand                       { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return EXPAND; }
volatile                     { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return VOLATILE; }
iteration                    { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return ITERATION; }


"="                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return EQUALS; }
"+"                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return PLUS; }
"-"                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return MINUS; }
"*"                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return TIMES; }
"*="                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return TIMESEQUAL; }
"+="                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return PLUSEQUAL; }
"-="                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return MINUSEQUAL; }

\(                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng);  return LPAREN; }
\)                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng);  return RPAREN; }
\{                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return LBRACE; }
\}                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return RBRACE; }
\[                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng);  return LBRACKET; }
\]                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return RBRACKET; }
,                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng);  return COMMA; }
:                        { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return COLON; }
;                        { tce_lineno = line;  tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return SEMI; }


-?[0-9]+|[0-9]*            { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return ICONST; }
-?[0-9]+"."[0-9]+            { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return FCONST; }
-?[0-9]+"/"[0-9]+            { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return FRACCONST; }
[A-Za-z\_][A-Za-z0-9\_]* { tce_lineno = line; tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return ID; }

%%


void yyerror(char *s)
{
 fprintf(stderr,"%s at line: %d col: %d\n",s,line,tce_tokPos);
 exit(2);
}



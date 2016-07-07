%{
#include "parser.h"
#include "error.h"

char *copy(char *s, int len) {
  char *d = (char *)calloc(sizeof(char),len+1);
  strcpy(d,s);
  return d;
}

int line = 1, col = 1;
//FIXME: Column nos reported are inaccurate
%}


%option yylineno
%option reentrant noyywrap
%option extra-type="char *"
%option outfile="scanner.c" header-file="scanner.h"

%%

 /* Ignore single-line comments and whitespace. */
"//".*                    { ; }
[ \t]+                  { col+=yyleng;}
\n                      {col = 1; ++line;}

range                        { tce_tokPos = col; col += (yyleng); yyextra = copy(yytext,yyleng); return RANGE; }
index                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return INDEX; }
array                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return ARRAY; }
expand                       { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return EXPAND; }
volatile                     { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return VOLATILE; }
iteration                    { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return ITERATION; }


"="                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return EQUALS; }
"+"                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return PLUS; }
"-"                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return MINUS; }
"*"                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return TIMES; }
"*="                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return TIMESEQUAL; }
"+="                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return PLUSEQUAL; }
"-="                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return MINUSEQUAL; }

\(                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng);  return LPAREN; }
\)                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng);  return RPAREN; }
\{                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return LBRACE; }
\}                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return RBRACE; }
\[                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng);  return LBRACKET; }
\]                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return RBRACKET; }
,                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng);  return COMMA; }
:                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return COLON; }
;                        { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return SEMI; }


[0-9]+|[0-9]*            { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return ICONST; }
[0-9]+"."[0-9]+            { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return FCONST; }
[A-Za-z\_][A-Za-z0-9\_]* { tce_tokPos = col; col += (yyleng);yyextra = copy(yytext,yyleng); return ID; }

%%


void yyerror(char *s)
{
 printf("%s at line: %d col: %d\n",s,line,tce_tokPos);
}



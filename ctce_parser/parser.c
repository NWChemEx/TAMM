/*
** 2000-05-29
**
** The author disclaims copyright to this source code.  In place of
** a legal notice, here is a blessing:
**
**    May you do good and not evil.
**    May you find forgiveness for yourself and forgive others.
**    May you share freely, never taking more than you give.
**
*************************************************************************
** Driver template for the LEMON parser generator.
**
** The "lemon" program processes an LALR(1) input grammar file, then uses
** this template to construct a parser.  The "lemon" program inserts text
** at each "%%" line.  Also, any "P-a-r-s-e" identifer prefix (without the
** interstitial "-" characters) contained in this template is changed into
** the value of the %name directive from the grammar.  Otherwise, the content
** of this template is copied straight through into the generate parser
** source file.
**
** The following is the concatenation of all %include directives from the
** input grammar file:
*/
#include <stdio.h>
/************ Begin %include sections from the grammar ************************/
#line 1 "parser.y"


    #include "parser.h"
    #include "absyn.h"
    #include "error.h"
    
    void yyerror(char *s);

    
#line 38 "parser.c"
/**************** End of %include directives **********************************/
/* These constants specify the various numeric values for terminal symbols
** in a format understandable to "makeheaders".  This section is blank unless
** "lemon" is run with the "-m" command-line option.
***************** Begin makeheaders token definitions *************************/
/**************** End makeheaders token definitions ***************************/

/* The next sections is a series of control #defines.
** various aspects of the generated parser.
**    YYCODETYPE         is the data type used to store the integer codes
**                       that represent terminal and non-terminal symbols.
**                       "unsigned char" is used if there are fewer than
**                       256 symbols.  Larger types otherwise.
**    YYNOCODE           is a number of type YYCODETYPE that is not used for
**                       any terminal or nonterminal symbol.
**    YYFALLBACK         If defined, this indicates that one or more tokens
**                       (also known as: "terminal symbols") have fall-back
**                       values which should be used if the original symbol
**                       would not parse.  This permits keywords to sometimes
**                       be used as identifiers, for example.
**    YYACTIONTYPE       is the data type used for "action codes" - numbers
**                       that indicate what to do in response to the next
**                       token.
**    ParseTOKENTYPE     is the data type used for minor type for terminal
**                       symbols.  Background: A "minor type" is a semantic
**                       value associated with a terminal or non-terminal
**                       symbols.  For example, for an "ID" terminal symbol,
**                       the minor type might be the name of the identifier.
**                       Each non-terminal can have a different minor type.
**                       Terminal symbols all have the same minor type, though.
**                       This macros defines the minor type for terminal 
**                       symbols.
**    YYMINORTYPE        is the data type used for all minor types.
**                       This is typically a union of many types, one of
**                       which is ParseTOKENTYPE.  The entry in the union
**                       for terminal symbols is called "yy0".
**    YYSTACKDEPTH       is the maximum depth of the parser's stack.  If
**                       zero the stack is dynamically sized using realloc()
**    ParseARG_SDECL     A static variable declaration for the %extra_argument
**    ParseARG_PDECL     A parameter declaration for the %extra_argument
**    ParseARG_STORE     Code to store %extra_argument into yypParser
**    ParseARG_FETCH     Code to extract %extra_argument from yypParser
**    YYERRORSYMBOL      is the code number of the error symbol.  If not
**                       defined, then do no error processing.
**    YYNSTATE           the combined number of states.
**    YYNRULE            the number of rules in the grammar
**    YY_MAX_SHIFT       Maximum value for shift actions
**    YY_MIN_SHIFTREDUCE Minimum value for shift-reduce actions
**    YY_MAX_SHIFTREDUCE Maximum value for shift-reduce actions
**    YY_MIN_REDUCE      Maximum value for reduce actions
**    YY_ERROR_ACTION    The yy_action[] code for syntax error
**    YY_ACCEPT_ACTION   The yy_action[] code for accept
**    YY_NO_ACTION       The yy_action[] code for no-op
*/
#ifndef INTERFACE
# define INTERFACE 1
#endif
/************* Begin control #defines *****************************************/
#define YYCODETYPE unsigned char
#define YYNOCODE 61
#define YYACTIONTYPE unsigned char
#define ParseTOKENTYPE void*
typedef union {
  int yyinit;
  ParseTOKENTYPE yy0;
} YYMINORTYPE;
#ifndef YYSTACKDEPTH
#define YYSTACKDEPTH 100
#endif
#define ParseARG_SDECL  TranslationUnit* root ;
#define ParseARG_PDECL , TranslationUnit* root 
#define ParseARG_FETCH  TranslationUnit* root  = yypParser->root 
#define ParseARG_STORE yypParser->root  = root 
#define YYNSTATE             54
#define YYNRULE              61
#define YY_MAX_SHIFT         53
#define YY_MIN_SHIFTREDUCE   104
#define YY_MAX_SHIFTREDUCE   164
#define YY_MIN_REDUCE        165
#define YY_MAX_REDUCE        225
#define YY_ERROR_ACTION      226
#define YY_ACCEPT_ACTION     227
#define YY_NO_ACTION         228
/************* End control #defines *******************************************/

/* Define the yytestcase() macro to be a no-op if is not already defined
** otherwise.
**
** Applications can choose to define yytestcase() in the %include section
** to a macro that can assist in verifying code coverage.  For production
** code the yytestcase() macro should be turned off.  But it is useful
** for testing.
*/
#ifndef yytestcase
# define yytestcase(X)
#endif


/* Next are the tables used to determine what action to take based on the
** current state and lookahead token.  These tables are used to implement
** functions that take a state number and lookahead value and return an
** action integer.  
**
** Suppose the action integer is N.  Then the action is determined as
** follows
**
**   0 <= N <= YY_MAX_SHIFT             Shift N.  That is, push the lookahead
**                                      token onto the stack and goto state N.
**
**   N between YY_MIN_SHIFTREDUCE       Shift to an arbitrary state then
**     and YY_MAX_SHIFTREDUCE           reduce by rule N-YY_MIN_SHIFTREDUCE.
**
**   N between YY_MIN_REDUCE            Reduce by rule N-YY_MIN_REDUCE
**     and YY_MAX_REDUCE

**   N == YY_ERROR_ACTION               A syntax error has occurred.
**
**   N == YY_ACCEPT_ACTION              The parser accepts its input.
**
**   N == YY_NO_ACTION                  No such action.  Denotes unused
**                                      slots in the yy_action[] table.
**
** The action table is constructed as a single large table named yy_action[].
** Given state S and lookahead X, the action is computed as
**
**      yy_action[ yy_shift_ofst[S] + X ]
**
** If the index value yy_shift_ofst[S]+X is out of range or if the value
** yy_lookahead[yy_shift_ofst[S]+X] is not equal to X or if yy_shift_ofst[S]
** is equal to YY_SHIFT_USE_DFLT, it means that the action is not in the table
** and that yy_default[S] should be used instead.  
**
** The formula above is for computing the action when the lookahead is
** a terminal symbol.  If the lookahead is a non-terminal (as occurs after
** a reduce action) then the yy_reduce_ofst[] array is used in place of
** the yy_shift_ofst[] array and YY_REDUCE_USE_DFLT is used in place of
** YY_SHIFT_USE_DFLT.
**
** The following are the tables generated in this section:
**
**  yy_action[]        A single table containing all actions.
**  yy_lookahead[]     A table containing the lookahead for each entry in
**                     yy_action.  Used to detect hash collisions.
**  yy_shift_ofst[]    For each state, the offset into yy_action for
**                     shifting terminals.
**  yy_reduce_ofst[]   For each state, the offset into yy_action for
**                     shifting non-terminals after a reduce.
**  yy_default[]       Default action for each state.
**
*********** Begin parsing tables **********************************************/
#define YY_ACTTAB_COUNT (161)
static const YYACTIONTYPE yy_action[] = {
 /*     0 */   109,  110,  111,  112,  113,  114,  115,  116,  117,  227,
 /*    10 */    13,   28,  153,  125,  126,  127,  134,   20,  120,  154,
 /*    20 */    29,   24,   42,  148,  150,    7,    6,  124,  120,  107,
 /*    30 */    31,   34,  125,  126,  127,   19,   22,   18,   17,   46,
 /*    40 */    49,   29,    3,  153,   15,   14,   40,  135,   20,  161,
 /*    50 */   154,  142,   24,   42,  148,  150,    7,    6,  137,  138,
 /*    60 */   139,  140,   44,  125,  126,  127,   50,  153,   35,  122,
 /*    70 */   106,   53,   41,    3,  154,  153,   24,   42,  148,  150,
 /*    80 */    45,  153,  154,  153,   24,   42,  148,  150,  154,  153,
 /*    90 */   154,   43,  148,  150,  152,  150,  154,  158,  153,  120,
 /*   100 */   151,  150,  165,    4,  120,  154,   25,   25,   11,  149,
 /*   110 */   150,   32,  120,  124,  120,   48,   49,   36,  131,  120,
 /*   120 */   143,  144,   29,   29,   33,   16,   37,    2,  163,  162,
 /*   130 */    27,   38,   23,    5,   39,  123,  130,   52,   11,   12,
 /*   140 */   132,   29,  133,  160,  164,   51,   21,  155,  121,    8,
 /*   150 */   136,    1,   16,   29,   30,   26,    9,  129,   47,   10,
 /*   160 */   128,
};
static const YYCODETYPE yy_lookahead[] = {
 /*     0 */    33,   34,   35,   36,   37,   38,   39,   40,   41,   28,
 /*    10 */    29,    1,   45,   12,   13,   14,   49,   50,   31,   52,
 /*    20 */    10,   54,   55,   56,   57,    5,    6,   11,   31,    9,
 /*    30 */    43,   11,   12,   13,   14,   15,   10,   17,   18,   42,
 /*    40 */    43,   10,   22,   45,   24,   25,   26,   49,   50,   23,
 /*    50 */    52,   20,   54,   55,   56,   57,    5,    6,    1,    2,
 /*    60 */     3,    4,   11,   12,   13,   14,   11,   45,   44,   45,
 /*    70 */    30,   31,   50,   22,   52,   45,   54,   55,   56,   57,
 /*    80 */    50,   45,   52,   45,   54,   55,   56,   57,   52,   45,
 /*    90 */    52,   55,   56,   57,   56,   57,   52,   11,   45,   31,
 /*   100 */    56,   57,    0,   51,   31,   52,   58,   59,   22,   56,
 /*   110 */    57,   43,   31,   11,   31,   42,   43,   46,   47,   31,
 /*   120 */     5,    6,   10,   10,   43,   19,   43,   21,   16,   16,
 /*   130 */    10,   43,    1,   53,   45,   45,   16,   45,   22,   21,
 /*   140 */    47,   10,   48,   59,   16,   31,    1,   23,   31,    7,
 /*   150 */    16,   32,   19,   10,    8,   20,   19,   16,   20,   19,
 /*   160 */    16,
};
#define YY_SHIFT_USE_DFLT (-1)
#define YY_SHIFT_COUNT (53)
#define YY_SHIFT_MIN   (0)
#define YY_SHIFT_MAX   (146)
static const short yy_shift_ofst[] = {
 /*     0 */    -1,   20,   51,   51,   51,   51,   51,   51,   51,   16,
 /*    10 */    16,    1,   86,  102,   16,   16,   16,   55,   16,   16,
 /*    20 */    57,    1,    1,    1,  115,  116,  118,   55,   16,   16,
 /*    30 */    -1,  112,  113,   31,  106,   26,  120,   10,  131,  128,
 /*    40 */   145,  124,  142,  142,  133,  134,  135,  137,  138,  143,
 /*    50 */   140,  141,  144,  146,
};
#define YY_REDUCE_USE_DFLT (-34)
#define YY_REDUCE_COUNT (30)
#define YY_REDUCE_MIN   (-33)
#define YY_REDUCE_MAX   (119)
static const signed char yy_reduce_ofst[] = {
 /*     0 */   -19,  -33,   -2,   22,   30,   36,   38,   44,   53,   -3,
 /*    10 */    73,   24,   48,   40,  -13,   68,   81,   71,   83,   88,
 /*    20 */    52,   89,   90,   92,   80,   84,   94,   93,  114,  117,
 /*    30 */   119,
};
static const YYACTIONTYPE yy_default[] = {
 /*     0 */   166,  226,  226,  226,  226,  226,  226,  226,  226,  179,
 /*    10 */   179,  226,  226,  226,  226,  226,  226,  226,  226,  226,
 /*    20 */   226,  226,  226,  226,  206,  218,  217,  226,  226,  226,
 /*    30 */   169,  226,  226,  226,  202,  226,  226,  226,  226,  226,
 /*    40 */   226,  226,  207,  208,  202,  226,  226,  226,  226,  180,
 /*    50 */   226,  226,  226,  226,
};
/********** End of lemon-generated parsing tables *****************************/

/* The next table maps tokens (terminal symbols) into fallback tokens.  
** If a construct like the following:
** 
**      %fallback ID X Y Z.
**
** appears in the grammar, then ID becomes a fallback token for X, Y,
** and Z.  Whenever one of the tokens X, Y, or Z is input to the parser
** but it does not parse, the type of the token is changed to ID and
** the parse is retried before an error is thrown.
**
** This feature can be used, for example, to cause some keywords in a language
** to revert to identifiers if they keyword does not apply in the context where
** it appears.
*/
#ifdef YYFALLBACK
static const YYCODETYPE yyFallback[] = {
};
#endif /* YYFALLBACK */

/* The following structure represents a single element of the
** parser's stack.  Information stored includes:
**
**   +  The state number for the parser at this level of the stack.
**
**   +  The value of the token stored at this level of the stack.
**      (In other words, the "major" token.)
**
**   +  The semantic value stored at this level of the stack.  This is
**      the information used by the action routines in the grammar.
**      It is sometimes called the "minor" token.
**
** After the "shift" half of a SHIFTREDUCE action, the stateno field
** actually contains the reduce action for the second half of the
** SHIFTREDUCE.
*/
struct yyStackEntry {
  YYACTIONTYPE stateno;  /* The state-number, or reduce action in SHIFTREDUCE */
  YYCODETYPE major;      /* The major token value.  This is the code
                         ** number for the token at this stack level */
  YYMINORTYPE minor;     /* The user-supplied minor token value.  This
                         ** is the value of the token  */
};
typedef struct yyStackEntry yyStackEntry;

/* The state of the parser is completely contained in an instance of
** the following structure */
struct yyParser {
  yyStackEntry *yytos;          /* Pointer to top element of the stack */
#ifdef YYTRACKMAXSTACKDEPTH
  int yyhwm;                    /* High-water mark of the stack */
#endif
#ifndef YYNOERRORRECOVERY
  int yyerrcnt;                 /* Shifts left before out of the error */
#endif
  ParseARG_SDECL                /* A place to hold %extra_argument */
#if YYSTACKDEPTH<=0
  int yystksz;                  /* Current side of the stack */
  yyStackEntry *yystack;        /* The parser's stack */
  yyStackEntry yystk0;          /* First stack entry */
#else
  yyStackEntry yystack[YYSTACKDEPTH];  /* The parser's stack */
#endif
};
typedef struct yyParser yyParser;

#ifndef NDEBUG
#include <stdio.h>
static FILE *yyTraceFILE = 0;
static char *yyTracePrompt = 0;
#endif /* NDEBUG */

#ifndef NDEBUG
/* 
** Turn parser tracing on by giving a stream to which to write the trace
** and a prompt to preface each trace message.  Tracing is turned off
** by making either argument NULL 
**
** Inputs:
** <ul>
** <li> A FILE* to which trace output should be written.
**      If NULL, then tracing is turned off.
** <li> A prefix string written at the beginning of every
**      line of trace output.  If NULL, then tracing is
**      turned off.
** </ul>
**
** Outputs:
** None.
*/
void ParseTrace(FILE *TraceFILE, char *zTracePrompt){
  yyTraceFILE = TraceFILE;
  yyTracePrompt = zTracePrompt;
  if( yyTraceFILE==0 ) yyTracePrompt = 0;
  else if( yyTracePrompt==0 ) yyTraceFILE = 0;
}
#endif /* NDEBUG */

#ifndef NDEBUG
/* For tracing shifts, the names of all terminals and nonterminals
** are required.  The following table supplies these names */
static const char *const yyTokenName[] = { 
  "$",             "EQUALS",        "TIMESEQUAL",    "PLUSEQUAL",   
  "MINUSEQUAL",    "PLUS",          "MINUS",         "TIMES",       
  "LBRACE",        "RBRACE",        "COMMA",         "ID",          
  "ICONST",        "FCONST",        "FRACCONST",     "RANGE",       
  "SEMI",          "INDEX",         "ARRAY",         "LBRACKET",    
  "RBRACKET",      "COLON",         "LPAREN",        "RPAREN",      
  "EXPAND",        "VOLATILE",      "ITERATION",     "error",       
  "translation_unit",  "compound_element_list",  "compound_element",  "identifier",  
  "element_list",  "element",       "declaration",   "statement",   
  "range_declaration",  "index_declaration",  "array_declaration",  "expansion_declaration",
  "volatile_declaration",  "iteration_declaration",  "id_list_opt",   "id_list",     
  "num_list",      "numerical_constant",  "array_structure_list",  "array_structure",
  "permut_symmetry_opt",  "assignment_statement",  "expression",    "assignment_operator",
  "array_reference",  "plusORMinus",   "additive_expression",  "multiplicative_expression",
  "unary_expression",  "primary_expression",  "symmetry_group_list",  "symmetry_group",
};
#endif /* NDEBUG */

#ifndef NDEBUG
/* For tracing reduce actions, the names of all rules are required.
*/
static const char *const yyRuleName[] = {
 /*   0 */ "translation_unit ::= compound_element_list",
 /*   1 */ "compound_element_list ::=",
 /*   2 */ "compound_element_list ::= compound_element_list compound_element",
 /*   3 */ "compound_element ::= identifier LBRACE element_list RBRACE",
 /*   4 */ "element_list ::=",
 /*   5 */ "element_list ::= element_list element",
 /*   6 */ "element ::= declaration",
 /*   7 */ "element ::= statement",
 /*   8 */ "declaration ::= range_declaration",
 /*   9 */ "declaration ::= index_declaration",
 /*  10 */ "declaration ::= array_declaration",
 /*  11 */ "declaration ::= expansion_declaration",
 /*  12 */ "declaration ::= volatile_declaration",
 /*  13 */ "declaration ::= iteration_declaration",
 /*  14 */ "id_list_opt ::=",
 /*  15 */ "id_list_opt ::= id_list",
 /*  16 */ "id_list ::= identifier",
 /*  17 */ "id_list ::= id_list COMMA identifier",
 /*  18 */ "num_list ::= numerical_constant",
 /*  19 */ "num_list ::= num_list COMMA numerical_constant",
 /*  20 */ "identifier ::= ID",
 /*  21 */ "numerical_constant ::= ICONST",
 /*  22 */ "numerical_constant ::= FCONST",
 /*  23 */ "numerical_constant ::= FRACCONST",
 /*  24 */ "range_declaration ::= RANGE id_list EQUALS numerical_constant SEMI",
 /*  25 */ "index_declaration ::= INDEX id_list EQUALS identifier SEMI",
 /*  26 */ "array_declaration ::= ARRAY array_structure_list SEMI",
 /*  27 */ "array_structure_list ::= array_structure",
 /*  28 */ "array_structure_list ::= array_structure_list COMMA array_structure",
 /*  29 */ "array_structure ::= ID LBRACKET id_list_opt RBRACKET LBRACKET id_list_opt RBRACKET permut_symmetry_opt",
 /*  30 */ "statement ::= assignment_statement",
 /*  31 */ "statement ::= ID COLON assignment_statement",
 /*  32 */ "assignment_statement ::= expression assignment_operator expression SEMI",
 /*  33 */ "assignment_operator ::= EQUALS",
 /*  34 */ "assignment_operator ::= TIMESEQUAL",
 /*  35 */ "assignment_operator ::= PLUSEQUAL",
 /*  36 */ "assignment_operator ::= MINUSEQUAL",
 /*  37 */ "array_reference ::= ID",
 /*  38 */ "array_reference ::= ID LBRACKET id_list RBRACKET",
 /*  39 */ "plusORMinus ::= PLUS",
 /*  40 */ "plusORMinus ::= MINUS",
 /*  41 */ "expression ::= additive_expression",
 /*  42 */ "additive_expression ::= multiplicative_expression",
 /*  43 */ "additive_expression ::= additive_expression plusORMinus multiplicative_expression",
 /*  44 */ "multiplicative_expression ::= unary_expression",
 /*  45 */ "multiplicative_expression ::= multiplicative_expression TIMES unary_expression",
 /*  46 */ "unary_expression ::= primary_expression",
 /*  47 */ "unary_expression ::= PLUS unary_expression",
 /*  48 */ "unary_expression ::= MINUS unary_expression",
 /*  49 */ "primary_expression ::= numerical_constant",
 /*  50 */ "primary_expression ::= array_reference",
 /*  51 */ "primary_expression ::= LPAREN expression RPAREN",
 /*  52 */ "permut_symmetry_opt ::=",
 /*  53 */ "permut_symmetry_opt ::= COLON symmetry_group_list",
 /*  54 */ "permut_symmetry_opt ::= COLON ID",
 /*  55 */ "symmetry_group_list ::= symmetry_group",
 /*  56 */ "symmetry_group_list ::= symmetry_group_list symmetry_group",
 /*  57 */ "symmetry_group ::= LPAREN num_list RPAREN",
 /*  58 */ "expansion_declaration ::= EXPAND id_list SEMI",
 /*  59 */ "volatile_declaration ::= VOLATILE id_list SEMI",
 /*  60 */ "iteration_declaration ::= ITERATION EQUALS numerical_constant SEMI",
};
#endif /* NDEBUG */


#if YYSTACKDEPTH<=0
/*
** Try to increase the size of the parser stack.  Return the number
** of errors.  Return 0 on success.
*/
static int yyGrowStack(yyParser *p){
  int newSize;
  int idx;
  yyStackEntry *pNew;

  newSize = p->yystksz*2 + 100;
  idx = p->yytos ? (int)(p->yytos - p->yystack) : 0;
  if( p->yystack==&p->yystk0 ){
    pNew = malloc(newSize*sizeof(pNew[0]));
    if( pNew ) pNew[0] = p->yystk0;
  }else{
    pNew = realloc(p->yystack, newSize*sizeof(pNew[0]));
  }
  if( pNew ){
    p->yystack = pNew;
    p->yytos = &p->yystack[idx];
#ifndef NDEBUG
    if( yyTraceFILE ){
      fprintf(yyTraceFILE,"%sStack grows from %d to %d entries.\n",
              yyTracePrompt, p->yystksz, newSize);
    }
#endif
    p->yystksz = newSize;
  }
  return pNew==0; 
}
#endif

/* Datatype of the argument to the memory allocated passed as the
** second argument to ParseAlloc() below.  This can be changed by
** putting an appropriate #define in the %include section of the input
** grammar.
*/
#ifndef YYMALLOCARGTYPE
# define YYMALLOCARGTYPE size_t
#endif

/* 
** This function allocates a new parser.
** The only argument is a pointer to a function which works like
** malloc.
**
** Inputs:
** A pointer to the function used to allocate memory.
**
** Outputs:
** A pointer to a parser.  This pointer is used in subsequent calls
** to Parse and ParseFree.
*/
void *ParseAlloc(void *(*mallocProc)(YYMALLOCARGTYPE)){
  yyParser *pParser;
  pParser = (yyParser*)(*mallocProc)( (YYMALLOCARGTYPE)sizeof(yyParser) );
  if( pParser ){
#ifdef YYTRACKMAXSTACKDEPTH
    pParser->yyhwm = 0;
#endif
#if YYSTACKDEPTH<=0
    pParser->yytos = NULL;
    pParser->yystack = NULL;
    pParser->yystksz = 0;
    if( yyGrowStack(pParser) ){
      pParser->yystack = &pParser->yystk0;
      pParser->yystksz = 1;
    }
#endif
#ifndef YYNOERRORRECOVERY
    pParser->yyerrcnt = -1;
#endif
    pParser->yytos = pParser->yystack;
    pParser->yystack[0].stateno = 0;
    pParser->yystack[0].major = 0;
  }
  return pParser;
}

/* The following function deletes the "minor type" or semantic value
** associated with a symbol.  The symbol can be either a terminal
** or nonterminal. "yymajor" is the symbol code, and "yypminor" is
** a pointer to the value to be deleted.  The code used to do the 
** deletions is derived from the %destructor and/or %token_destructor
** directives of the input grammar.
*/
static void yy_destructor(
  yyParser *yypParser,    /* The parser */
  YYCODETYPE yymajor,     /* Type code for object to destroy */
  YYMINORTYPE *yypminor   /* The object to be destroyed */
){
  ParseARG_FETCH;
  switch( yymajor ){
    /* Here is inserted the actions which take place when a
    ** terminal or non-terminal is destroyed.  This can happen
    ** when the symbol is popped from the stack during a
    ** reduce or during error processing or when a parser is 
    ** being destroyed before it is finished parsing.
    **
    ** Note: during a reduce, the only symbols destroyed are those
    ** which appear on the RHS of the rule, but which are *not* used
    ** inside the C code.
    */
/********* Begin destructor definitions ***************************************/
      /* TERMINAL Destructor */
    case 1: /* EQUALS */
    case 2: /* TIMESEQUAL */
    case 3: /* PLUSEQUAL */
    case 4: /* MINUSEQUAL */
    case 5: /* PLUS */
    case 6: /* MINUS */
    case 7: /* TIMES */
    case 8: /* LBRACE */
    case 9: /* RBRACE */
    case 10: /* COMMA */
    case 11: /* ID */
    case 12: /* ICONST */
    case 13: /* FCONST */
    case 14: /* FRACCONST */
    case 15: /* RANGE */
    case 16: /* SEMI */
    case 17: /* INDEX */
    case 18: /* ARRAY */
    case 19: /* LBRACKET */
    case 20: /* RBRACKET */
    case 21: /* COLON */
    case 22: /* LPAREN */
    case 23: /* RPAREN */
    case 24: /* EXPAND */
    case 25: /* VOLATILE */
    case 26: /* ITERATION */
{
#line 16 "parser.y"
 free((yypminor->yy0));
#line 583 "parser.c"
}
      break;
/********* End destructor definitions *****************************************/
    default:  break;   /* If no destructor action specified: do nothing */
  }
}

/*
** Pop the parser's stack once.
**
** If there is a destructor routine associated with the token which
** is popped from the stack, then call it.
*/
static void yy_pop_parser_stack(yyParser *pParser){
  yyStackEntry *yytos;
  assert( pParser->yytos!=0 );
  assert( pParser->yytos > pParser->yystack );
  yytos = pParser->yytos--;
#ifndef NDEBUG
  if( yyTraceFILE ){
    fprintf(yyTraceFILE,"%sPopping %s\n",
      yyTracePrompt,
      yyTokenName[yytos->major]);
  }
#endif
  yy_destructor(pParser, yytos->major, &yytos->minor);
}

/* 
** Deallocate and destroy a parser.  Destructors are called for
** all stack elements before shutting the parser down.
**
** If the YYPARSEFREENEVERNULL macro exists (for example because it
** is defined in a %include section of the input grammar) then it is
** assumed that the input pointer is never NULL.
*/
void ParseFree(
  void *p,                    /* The parser to be deleted */
  void (*freeProc)(void*)     /* Function used to reclaim memory */
){
  yyParser *pParser = (yyParser*)p;
#ifndef YYPARSEFREENEVERNULL
  if( pParser==0 ) return;
#endif
  while( pParser->yytos>pParser->yystack ) yy_pop_parser_stack(pParser);
#if YYSTACKDEPTH<=0
  if( pParser->yystack!=&pParser->yystk0 ) free(pParser->yystack);
#endif
  (*freeProc)((void*)pParser);
}

/*
** Return the peak depth of the stack for a parser.
*/
#ifdef YYTRACKMAXSTACKDEPTH
int ParseStackPeak(void *p){
  yyParser *pParser = (yyParser*)p;
  return pParser->yyhwm;
}
#endif

/*
** Find the appropriate action for a parser given the terminal
** look-ahead token iLookAhead.
*/
static unsigned int yy_find_shift_action(
  yyParser *pParser,        /* The parser */
  YYCODETYPE iLookAhead     /* The look-ahead token */
){
  int i;
  int stateno = pParser->yytos->stateno;
 
  if( stateno>=YY_MIN_REDUCE ) return stateno;
  assert( stateno <= YY_SHIFT_COUNT );
  do{
    i = yy_shift_ofst[stateno];
    if( i==YY_SHIFT_USE_DFLT ) return yy_default[stateno];
    assert( iLookAhead!=YYNOCODE );
    i += iLookAhead;
    if( i<0 || i>=YY_ACTTAB_COUNT || yy_lookahead[i]!=iLookAhead ){
      if( iLookAhead>0 ){
#ifdef YYFALLBACK
        YYCODETYPE iFallback;            /* Fallback token */
        if( iLookAhead<sizeof(yyFallback)/sizeof(yyFallback[0])
               && (iFallback = yyFallback[iLookAhead])!=0 ){
#ifndef NDEBUG
          if( yyTraceFILE ){
            fprintf(yyTraceFILE, "%sFALLBACK %s => %s\n",
               yyTracePrompt, yyTokenName[iLookAhead], yyTokenName[iFallback]);
          }
#endif
          assert( yyFallback[iFallback]==0 ); /* Fallback loop must terminate */
          iLookAhead = iFallback;
          continue;
        }
#endif
#ifdef YYWILDCARD
        {
          int j = i - iLookAhead + YYWILDCARD;
          if( 
#if YY_SHIFT_MIN+YYWILDCARD<0
            j>=0 &&
#endif
#if YY_SHIFT_MAX+YYWILDCARD>=YY_ACTTAB_COUNT
            j<YY_ACTTAB_COUNT &&
#endif
            yy_lookahead[j]==YYWILDCARD
          ){
#ifndef NDEBUG
            if( yyTraceFILE ){
              fprintf(yyTraceFILE, "%sWILDCARD %s => %s\n",
                 yyTracePrompt, yyTokenName[iLookAhead],
                 yyTokenName[YYWILDCARD]);
            }
#endif /* NDEBUG */
            return yy_action[j];
          }
        }
#endif /* YYWILDCARD */
      }
      return yy_default[stateno];
    }else{
      return yy_action[i];
    }
  }while(1);
}

/*
** Find the appropriate action for a parser given the non-terminal
** look-ahead token iLookAhead.
*/
static int yy_find_reduce_action(
  int stateno,              /* Current state number */
  YYCODETYPE iLookAhead     /* The look-ahead token */
){
  int i;
#ifdef YYERRORSYMBOL
  if( stateno>YY_REDUCE_COUNT ){
    return yy_default[stateno];
  }
#else
  assert( stateno<=YY_REDUCE_COUNT );
#endif
  i = yy_reduce_ofst[stateno];
  assert( i!=YY_REDUCE_USE_DFLT );
  assert( iLookAhead!=YYNOCODE );
  i += iLookAhead;
#ifdef YYERRORSYMBOL
  if( i<0 || i>=YY_ACTTAB_COUNT || yy_lookahead[i]!=iLookAhead ){
    return yy_default[stateno];
  }
#else
  assert( i>=0 && i<YY_ACTTAB_COUNT );
  assert( yy_lookahead[i]==iLookAhead );
#endif
  return yy_action[i];
}

/*
** The following routine is called if the stack overflows.
*/
static void yyStackOverflow(yyParser *yypParser){
   ParseARG_FETCH;
   yypParser->yytos--;
#ifndef NDEBUG
   if( yyTraceFILE ){
     fprintf(yyTraceFILE,"%sStack Overflow!\n",yyTracePrompt);
   }
#endif
   while( yypParser->yytos>yypParser->yystack ) yy_pop_parser_stack(yypParser);
   /* Here code is inserted which will execute if the parser
   ** stack every overflows */
/******** Begin %stack_overflow code ******************************************/
/******** End %stack_overflow code ********************************************/
   ParseARG_STORE; /* Suppress warning about unused %extra_argument var */
}

/*
** Print tracing information for a SHIFT action
*/
#ifndef NDEBUG
static void yyTraceShift(yyParser *yypParser, int yyNewState){
  if( yyTraceFILE ){
    if( yyNewState<YYNSTATE ){
      fprintf(yyTraceFILE,"%sShift '%s', go to state %d\n",
         yyTracePrompt,yyTokenName[yypParser->yytos->major],
         yyNewState);
    }else{
      fprintf(yyTraceFILE,"%sShift '%s'\n",
         yyTracePrompt,yyTokenName[yypParser->yytos->major]);
    }
  }
}
#else
# define yyTraceShift(X,Y)
#endif

/*
** Perform a shift action.
*/
static void yy_shift(
  yyParser *yypParser,          /* The parser to be shifted */
  int yyNewState,               /* The new state to shift in */
  int yyMajor,                  /* The major token to shift in */
  ParseTOKENTYPE yyMinor        /* The minor token to shift in */
){
  yyStackEntry *yytos;
  yypParser->yytos++;
#ifdef YYTRACKMAXSTACKDEPTH
  if( (int)(yypParser->yytos - yypParser->yystack)>yypParser->yyhwm ){
    yypParser->yyhwm++;
    assert( yypParser->yyhwm == (int)(yypParser->yytos - yypParser->yystack) );
  }
#endif
#if YYSTACKDEPTH>0 
  if( yypParser->yytos>=&yypParser->yystack[YYSTACKDEPTH] ){
    yyStackOverflow(yypParser);
    return;
  }
#else
  if( yypParser->yytos>=&yypParser->yystack[yypParser->yystksz] ){
    if( yyGrowStack(yypParser) ){
      yyStackOverflow(yypParser);
      return;
    }
  }
#endif
  if( yyNewState > YY_MAX_SHIFT ){
    yyNewState += YY_MIN_REDUCE - YY_MIN_SHIFTREDUCE;
  }
  yytos = yypParser->yytos;
  yytos->stateno = (YYACTIONTYPE)yyNewState;
  yytos->major = (YYCODETYPE)yyMajor;
  yytos->minor.yy0 = yyMinor;
  yyTraceShift(yypParser, yyNewState);
}

/* The following table contains information about every rule that
** is used during the reduce.
*/
static const struct {
  YYCODETYPE lhs;         /* Symbol on the left-hand side of the rule */
  unsigned char nrhs;     /* Number of right-hand side symbols in the rule */
} yyRuleInfo[] = {
  { 28, 1 },
  { 29, 0 },
  { 29, 2 },
  { 30, 4 },
  { 32, 0 },
  { 32, 2 },
  { 33, 1 },
  { 33, 1 },
  { 34, 1 },
  { 34, 1 },
  { 34, 1 },
  { 34, 1 },
  { 34, 1 },
  { 34, 1 },
  { 42, 0 },
  { 42, 1 },
  { 43, 1 },
  { 43, 3 },
  { 44, 1 },
  { 44, 3 },
  { 31, 1 },
  { 45, 1 },
  { 45, 1 },
  { 45, 1 },
  { 36, 5 },
  { 37, 5 },
  { 38, 3 },
  { 46, 1 },
  { 46, 3 },
  { 47, 8 },
  { 35, 1 },
  { 35, 3 },
  { 49, 4 },
  { 51, 1 },
  { 51, 1 },
  { 51, 1 },
  { 51, 1 },
  { 52, 1 },
  { 52, 4 },
  { 53, 1 },
  { 53, 1 },
  { 50, 1 },
  { 54, 1 },
  { 54, 3 },
  { 55, 1 },
  { 55, 3 },
  { 56, 1 },
  { 56, 2 },
  { 56, 2 },
  { 57, 1 },
  { 57, 1 },
  { 57, 3 },
  { 48, 0 },
  { 48, 2 },
  { 48, 2 },
  { 58, 1 },
  { 58, 2 },
  { 59, 3 },
  { 39, 3 },
  { 40, 3 },
  { 41, 4 },
};

static void yy_accept(yyParser*);  /* Forward Declaration */

/*
** Perform a reduce action and the shift that must immediately
** follow the reduce.
*/
static void yy_reduce(
  yyParser *yypParser,         /* The parser */
  unsigned int yyruleno        /* Number of the rule by which to reduce */
){
  int yygoto;                     /* The next state */
  int yyact;                      /* The next action */
  yyStackEntry *yymsp;            /* The top of the parser's stack */
  int yysize;                     /* Amount to pop the stack */
  ParseARG_FETCH;
  yymsp = yypParser->yytos;
#ifndef NDEBUG
  if( yyTraceFILE && yyruleno<(int)(sizeof(yyRuleName)/sizeof(yyRuleName[0])) ){
    yysize = yyRuleInfo[yyruleno].nrhs;
    fprintf(yyTraceFILE, "%sReduce [%s], go to state %d.\n", yyTracePrompt,
      yyRuleName[yyruleno], yymsp[-yysize].stateno);
  }
#endif /* NDEBUG */

  /* Check that the stack is large enough to grow by a single entry
  ** if the RHS of the rule is empty.  This ensures that there is room
  ** enough on the stack to push the LHS value */
  if( yyRuleInfo[yyruleno].nrhs==0 ){
#ifdef YYTRACKMAXSTACKDEPTH
    if( (int)(yypParser->yytos - yypParser->yystack)>yypParser->yyhwm ){
      yypParser->yyhwm++;
      assert( yypParser->yyhwm == (int)(yypParser->yytos - yypParser->yystack));
    }
#endif
#if YYSTACKDEPTH>0 
    if( yypParser->yytos>=&yypParser->yystack[YYSTACKDEPTH-1] ){
      yyStackOverflow(yypParser);
      return;
    }
#else
    if( yypParser->yytos>=&yypParser->yystack[yypParser->yystksz-1] ){
      if( yyGrowStack(yypParser) ){
        yyStackOverflow(yypParser);
        return;
      }
      yymsp = yypParser->yytos;
    }
#endif
  }

  switch( yyruleno ){
  /* Beginning here are the reduction cases.  A typical example
  ** follows:
  **   case 0:
  **  #line <lineno> <grammarfile>
  **     { ... }           // User supplied code
  **  #line <lineno> <thisfile>
  **     break;
  */
/********** Begin reduce actions **********************************************/
        YYMINORTYPE yylhsminor;
      case 0: /* translation_unit ::= compound_element_list */
#line 29 "parser.y"
{
       *root = make_TranslationUnit(yymsp[0].minor.yy0);
    }
#line 957 "parser.c"
        break;
      case 1: /* compound_element_list ::= */
#line 35 "parser.y"
{
      yymsp[1].minor.yy0 = make_CompoundElemList(NULL,NULL);
    }
#line 964 "parser.c"
        break;
      case 2: /* compound_element_list ::= compound_element_list compound_element */
#line 39 "parser.y"
{
        addTail_CompoundElemList(yymsp[0].minor.yy0,yymsp[-1].minor.yy0);
        yylhsminor.yy0 = yymsp[-1].minor.yy0;
    }
#line 972 "parser.c"
  yymsp[-1].minor.yy0 = yylhsminor.yy0;
        break;
      case 3: /* compound_element ::= identifier LBRACE element_list RBRACE */
#line 46 "parser.y"
{
      CompoundElem ce = make_CompoundElem(yymsp[-1].minor.yy0);
      yymsp[-3].minor.yy0 = ce; 
    }
#line 981 "parser.c"
  yy_destructor(yypParser,8,&yymsp[-2].minor);
  yy_destructor(yypParser,9,&yymsp[0].minor);
        break;
      case 4: /* element_list ::= */
#line 52 "parser.y"
{ yymsp[1].minor.yy0 = make_ElemList(NULL,NULL); }
#line 988 "parser.c"
        break;
      case 5: /* element_list ::= element_list element */
#line 54 "parser.y"
{
       addTail_ElemList(yymsp[0].minor.yy0,yymsp[-1].minor.yy0);
       yylhsminor.yy0 = yymsp[-1].minor.yy0;
    }
#line 996 "parser.c"
  yymsp[-1].minor.yy0 = yylhsminor.yy0;
        break;
      case 6: /* element ::= declaration */
#line 62 "parser.y"
{ yylhsminor.yy0 = make_Elem_DeclList(yymsp[0].minor.yy0); }
#line 1002 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 7: /* element ::= statement */
#line 63 "parser.y"
{ yylhsminor.yy0 = make_Elem_Stmt(yymsp[0].minor.yy0); }
#line 1008 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 8: /* declaration ::= range_declaration */
      case 9: /* declaration ::= index_declaration */ yytestcase(yyruleno==9);
      case 10: /* declaration ::= array_declaration */ yytestcase(yyruleno==10);
      case 11: /* declaration ::= expansion_declaration */ yytestcase(yyruleno==11);
      case 12: /* declaration ::= volatile_declaration */ yytestcase(yyruleno==12);
      case 13: /* declaration ::= iteration_declaration */ yytestcase(yyruleno==13);
      case 15: /* id_list_opt ::= id_list */ yytestcase(yyruleno==15);
      case 33: /* assignment_operator ::= EQUALS */ yytestcase(yyruleno==33);
      case 34: /* assignment_operator ::= TIMESEQUAL */ yytestcase(yyruleno==34);
      case 35: /* assignment_operator ::= PLUSEQUAL */ yytestcase(yyruleno==35);
      case 36: /* assignment_operator ::= MINUSEQUAL */ yytestcase(yyruleno==36);
      case 39: /* plusORMinus ::= PLUS */ yytestcase(yyruleno==39);
      case 40: /* plusORMinus ::= MINUS */ yytestcase(yyruleno==40);
      case 42: /* additive_expression ::= multiplicative_expression */ yytestcase(yyruleno==42);
      case 44: /* multiplicative_expression ::= unary_expression */ yytestcase(yyruleno==44);
      case 46: /* unary_expression ::= primary_expression */ yytestcase(yyruleno==46);
      case 49: /* primary_expression ::= numerical_constant */ yytestcase(yyruleno==49);
      case 50: /* primary_expression ::= array_reference */ yytestcase(yyruleno==50);
#line 67 "parser.y"
{ yylhsminor.yy0 = yymsp[0].minor.yy0; }
#line 1031 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 14: /* id_list_opt ::= */
#line 76 "parser.y"
{ yymsp[1].minor.yy0 = NULL; }
#line 1037 "parser.c"
        break;
      case 16: /* id_list ::= identifier */
#line 80 "parser.y"
{ yylhsminor.yy0 = make_IDList(yymsp[0].minor.yy0,NULL); }
#line 1042 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 17: /* id_list ::= id_list COMMA identifier */
#line 81 "parser.y"
{ 
      addTail_IDList(yymsp[0].minor.yy0,yymsp[-2].minor.yy0);
      yylhsminor.yy0 = yymsp[-2].minor.yy0;
     }
#line 1051 "parser.c"
  yy_destructor(yypParser,10,&yymsp[-1].minor);
  yymsp[-2].minor.yy0 = yylhsminor.yy0;
        break;
      case 18: /* num_list ::= numerical_constant */
#line 87 "parser.y"
{ 
      ExpList e = make_ExpList(yymsp[0].minor.yy0,NULL); 
      yylhsminor.yy0 = e;
    }
#line 1061 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 19: /* num_list ::= num_list COMMA numerical_constant */
#line 92 "parser.y"
{ 
      addTail_ExpList(yymsp[0].minor.yy0,yymsp[-2].minor.yy0);
      yylhsminor.yy0 = yymsp[-2].minor.yy0;
     }
#line 1070 "parser.c"
  yy_destructor(yypParser,10,&yymsp[-1].minor);
  yymsp[-2].minor.yy0 = yylhsminor.yy0;
        break;
      case 20: /* identifier ::= ID */
#line 98 "parser.y"
{
      Identifier id = make_Identifier(tce_tokPos,yymsp[0].minor.yy0);
      id->lineno = tce_lineno;
      yylhsminor.yy0 = id;
    }
#line 1081 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 21: /* numerical_constant ::= ICONST */
#line 106 "parser.y"
{
      Exp e = make_NumConst(tce_tokPos,atoi(yymsp[0].minor.yy0));
      e->lineno = tce_lineno;
      yylhsminor.yy0 = e;
    }
#line 1091 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 22: /* numerical_constant ::= FCONST */
#line 112 "parser.y"
{
      Exp e = make_NumConst(tce_tokPos,atof(yymsp[0].minor.yy0));
      e->lineno = tce_lineno;
      yylhsminor.yy0 = e;      
    }
#line 1101 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 23: /* numerical_constant ::= FRACCONST */
#line 118 "parser.y"
{
      char* frac = yymsp[0].minor.yy0;
      char* str = strdup(frac);
      char* pch;
      char* nd[2];
      pch = strtok (str," /");
      int i =0;
      while (pch != NULL)
      {
        nd[i] = pch; i++;
        pch = strtok (NULL, " /");
        
      }
      float num = atof(nd[0]);
      float den = atof(nd[1]);
      //printf("%f/%f\n",num,den);
      Exp e = make_NumConst(tce_tokPos,num/den);
      e->lineno = tce_lineno;
      yylhsminor.yy0 = e;
    }
#line 1126 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 24: /* range_declaration ::= RANGE id_list EQUALS numerical_constant SEMI */
{  yy_destructor(yypParser,15,&yymsp[-4].minor);
#line 140 "parser.y"
{
      Exp e = yymsp[-1].minor.yy0;
      float val = -1;
      if(e->kind == is_NumConst) val = e->u.NumConst.value;
      //TODO: Error check for e's type

      IDList p = yymsp[-3].minor.yy0;
      DeclList dlist = make_DeclList(NULL,NULL);
      DeclList dl = dlist;  
      while(p != NULL){
        dl->head = make_RangeDecl(tce_tokPos,(p->head)->name,val);
        dl->head->lineno = (p->head)->lineno;
        p = p->tail;
        if(p!=NULL) {
          dl->tail = make_DeclList(NULL,NULL); 
          dl = dl->tail;
        }
      }

      dl = NULL;
      yymsp[-4].minor.yy0 = dlist;
    }
#line 1154 "parser.c"
  yy_destructor(yypParser,1,&yymsp[-2].minor);
  yy_destructor(yypParser,16,&yymsp[0].minor);
}
        break;
      case 25: /* index_declaration ::= INDEX id_list EQUALS identifier SEMI */
{  yy_destructor(yypParser,17,&yymsp[-4].minor);
#line 165 "parser.y"
{
      Identifier e = yymsp[-1].minor.yy0;

      IDList p = yymsp[-3].minor.yy0;
      DeclList dlist = make_DeclList(NULL,NULL);
      DeclList dl = dlist;  
      while(p != NULL){
        dl->head = make_IndexDecl(tce_tokPos,(p->head)->name,mkString(e->name));
        dl->head->lineno = (p->head)->lineno;
        p = p->tail;
        if(p!=NULL) {
          dl->tail = make_DeclList(NULL,NULL); 
          dl = dl->tail;
        }
      }
      dl = NULL;
      yymsp[-4].minor.yy0 = dlist;
    }
#line 1180 "parser.c"
  yy_destructor(yypParser,1,&yymsp[-2].minor);
  yy_destructor(yypParser,16,&yymsp[0].minor);
}
        break;
      case 26: /* array_declaration ::= ARRAY array_structure_list SEMI */
{  yy_destructor(yypParser,18,&yymsp[-2].minor);
#line 186 "parser.y"
{ yymsp[-2].minor.yy0 = yymsp[-1].minor.yy0; }
#line 1189 "parser.c"
  yy_destructor(yypParser,16,&yymsp[0].minor);
}
        break;
      case 27: /* array_structure_list ::= array_structure */
#line 187 "parser.y"
{ yylhsminor.yy0 = make_DeclList(yymsp[0].minor.yy0,NULL); }
#line 1196 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 28: /* array_structure_list ::= array_structure_list COMMA array_structure */
#line 188 "parser.y"
{ 
        addTail_DeclList(yymsp[0].minor.yy0,yymsp[-2].minor.yy0); 
        yylhsminor.yy0 = yymsp[-2].minor.yy0;
      }
#line 1205 "parser.c"
  yy_destructor(yypParser,10,&yymsp[-1].minor);
  yymsp[-2].minor.yy0 = yylhsminor.yy0;
        break;
      case 29: /* array_structure ::= ID LBRACKET id_list_opt RBRACKET LBRACKET id_list_opt RBRACKET permut_symmetry_opt */
#line 195 "parser.y"
{
     ctce_string id = yymsp[-7].minor.yy0;
     IDList p = yymsp[-5].minor.yy0;
     int countU = count_IDList(yymsp[-5].minor.yy0);
     
     p = yymsp[-5].minor.yy0;
     int ic = 0;
     ctce_string* indicesU = malloc(countU * sizeof(ctce_string));

     while (p!=NULL){
            indicesU[ic] = (p->head)->name;
            p = p->tail;
            ic++;
     } 

     int countL = count_IDList(yymsp[-2].minor.yy0);
     ic = 0;
     p = yymsp[-2].minor.yy0;
     ctce_string* indicesL = malloc(countL * sizeof(ctce_string));

     while (p!=NULL){
        indicesL[ic] = (p->head)->name;
        p = p->tail;
        ic++;
     } 

     Decl dec = make_ArrayDecl(tce_tokPos,id,indicesU,indicesL); 
     dec->u.ArrayDecl.ulen = countU;
     dec->u.ArrayDecl.llen = countL;
     dec->u.ArrayDecl.irrep = yymsp[0].minor.yy0; 
     dec->lineno = tce_lineno;
     yylhsminor.yy0 = dec;
     dec = NULL;
    }
#line 1245 "parser.c"
  yy_destructor(yypParser,19,&yymsp[-6].minor);
  yy_destructor(yypParser,20,&yymsp[-4].minor);
  yy_destructor(yypParser,19,&yymsp[-3].minor);
  yy_destructor(yypParser,20,&yymsp[-1].minor);
  yymsp[-7].minor.yy0 = yylhsminor.yy0;
        break;
      case 30: /* statement ::= assignment_statement */
#line 231 "parser.y"
{ 
      Stmt st = yymsp[0].minor.yy0; 
      st->u.AssignStmt.label = NULL; 
      yylhsminor.yy0=st;
    }
#line 1259 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 31: /* statement ::= ID COLON assignment_statement */
#line 237 "parser.y"
{ 
      Stmt st = yymsp[0].minor.yy0; 
      st->u.AssignStmt.label = yymsp[-2].minor.yy0;
      yylhsminor.yy0=st;
    }
#line 1269 "parser.c"
  yy_destructor(yypParser,21,&yymsp[-1].minor);
  yymsp[-2].minor.yy0 = yylhsminor.yy0;
        break;
      case 32: /* assignment_statement ::= expression assignment_operator expression SEMI */
#line 244 "parser.y"
{ 
        Stmt s;
        Exp lhs = yymsp[-3].minor.yy0;
        //lhs->lineno = tce_lineno;
        Exp rhs = yymsp[-1].minor.yy0;
        //rhs->lineno = tce_lineno;
        ctce_string oper = yymsp[-2].minor.yy0;

        if(strcmp(oper,"=")==0) { 
          s = make_AssignStmt(tce_tokPos,lhs,rhs); 
        }
        else {
          Exp tlhs = NULL;
          //if (lhs->kind == is_NumConst) tlhs = make_NumConst(tce_tokPos,0);
          //if (lhs->kind == is_Parenth)  tlhs = make_Parenth(tce_tokPos,NULL);
          
          //if (lhs->kind == is_Addition) tlhs = make_Addition(tce_tokPos,NULL);
          //if (lhs->kind == is_Multiplication) tlhs = make_Multiplication(tce_tokPos,NULL);
          
          if (lhs->kind == is_ArrayRef) {
            tlhs = make_Array(tce_tokPos,"",NULL); //create a copy of lhs for flattening
            //memcpy (tlhs, lhs, sizeof (lhs));
            tlhs->u.Array.name = mkString(lhs->u.Array.name);
            tlhs->u.Array.indices = mkIndexList(lhs->u.Array.indices,lhs->u.Array.length);
            tlhs->u.Array.length = lhs->u.Array.length;
            tlhs->lineno = lhs->lineno;
            tlhs->coef = lhs->coef;
          }
          assert(tlhs!=NULL); 

          Exp trhs = make_Parenth(tce_tokPos,rhs);
          //trhs->lineno = tce_lineno;

          if(strcmp(oper,"+=")==0) { 
            Exp tadd = make_Addition(tce_tokPos,make_ExpList(tlhs,make_ExpList(trhs,NULL)));        
            tadd->lineno = tlhs->lineno;
            s = make_AssignStmt(tce_tokPos,tlhs,tadd); 
          }
          else if(strcmp(oper,"-=")==0) { 
            trhs->coef *= -1;
            Exp tadd = make_Addition(tce_tokPos,make_ExpList(tlhs,make_ExpList(trhs,NULL)));   
            tadd->lineno = tlhs->lineno;           
            s = make_AssignStmt(tce_tokPos,tlhs,tadd); 
          }
            
          else if(strcmp(oper,"*=")==0) { 
            Exp tmult = make_Multiplication(tce_tokPos,make_ExpList(tlhs,make_ExpList(trhs,NULL)));      
            tmult->lineno = tlhs->lineno;     
            s = make_AssignStmt(tce_tokPos,tlhs,tmult); 
          }
        }
        s->u.AssignStmt.astype = oper;
        yylhsminor.yy0 = s;
      }
#line 1329 "parser.c"
  yy_destructor(yypParser,16,&yymsp[0].minor);
  yymsp[-3].minor.yy0 = yylhsminor.yy0;
        break;
      case 37: /* array_reference ::= ID */
#line 308 "parser.y"
{
      ctce_string id = yymsp[0].minor.yy0;
      Exp e = make_Array(tce_tokPos, id, NULL);
      e->lineno = tce_lineno;
      yylhsminor.yy0 = e;
    }
#line 1341 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 38: /* array_reference ::= ID LBRACKET id_list RBRACKET */
#line 315 "parser.y"
{
     ctce_string id = yymsp[-3].minor.yy0;

     IDList p = yymsp[-1].minor.yy0;
     int count = count_IDList(p);
     p = yymsp[-1].minor.yy0;

     ctce_string *indices = malloc(count * sizeof(ctce_string));

     int ic = 0;

      while (p!=NULL){
          indices[ic] = (p->head)->name;
          p = p->tail;
          ic++;
      } 

     Exp exp = make_Array(tce_tokPos, id, indices);
     exp->u.Array.length = count;
     exp->lineno = tce_lineno;
     yylhsminor.yy0 = exp;
     exp = NULL;
    }
#line 1369 "parser.c"
  yy_destructor(yypParser,19,&yymsp[-2].minor);
  yy_destructor(yypParser,20,&yymsp[0].minor);
  yymsp[-3].minor.yy0 = yylhsminor.yy0;
        break;
      case 41: /* expression ::= additive_expression */
#line 344 "parser.y"
{ 
      Exp e = yymsp[0].minor.yy0; 
      //e->lineno = tce_lineno;
      yylhsminor.yy0 = e;
    }
#line 1381 "parser.c"
  yymsp[0].minor.yy0 = yylhsminor.yy0;
        break;
      case 43: /* additive_expression ::= additive_expression plusORMinus multiplicative_expression */
#line 352 "parser.y"
{
     Exp e1 = yymsp[-2].minor.yy0;
     Exp e2 = yymsp[0].minor.yy0;
     ctce_string op = yymsp[-1].minor.yy0;
     ExpList el = make_ExpList(NULL,NULL);

     int clno = tce_lineno;
     if (e1->kind == is_Addition) e1->lineno = clno;

     if(strcmp(op,"-")==0) e2->coef *= -1;
     addTail_ExpList(e1,el);
     addTail_ExpList(e2,el);
     /*if (e1->kind == is_Addition) {
        addTail_ExpList(e1,el);
     }
     if (e2->kind == is_Addition) {
      addTail_ExpList(e2,el);
     }*/

     Exp nadd = make_Addition(tce_tokPos,el); 
     nadd->lineno = clno;
     yylhsminor.yy0 = nadd;
    }
#line 1409 "parser.c"
  yymsp[-2].minor.yy0 = yylhsminor.yy0;
        break;
      case 45: /* multiplicative_expression ::= multiplicative_expression TIMES unary_expression */
#line 379 "parser.y"
{
      Exp e1 = yymsp[-2].minor.yy0;
      Exp e2 = yymsp[0].minor.yy0;

      ExpList el = make_ExpList(NULL,NULL);
      float coef = 1;
      int clno = tce_lineno;
      if (e1->kind == is_Multiplication) {
        clno = e1->lineno;
        coef *= e1->coef;
     }
     addTail_ExpList(e1,el);
     if (e2->kind == is_Multiplication) {
      coef *= e2->coef;
     }
     addTail_ExpList(e2,el);

      Exp nmult = make_Multiplication(tce_tokPos,el); 
      nmult->coef = coef;
      nmult->lineno = clno;
      yylhsminor.yy0 = nmult;
    }
#line 1436 "parser.c"
  yy_destructor(yypParser,7,&yymsp[-1].minor);
  yymsp[-2].minor.yy0 = yylhsminor.yy0;
        break;
      case 47: /* unary_expression ::= PLUS unary_expression */
{  yy_destructor(yypParser,5,&yymsp[-1].minor);
#line 404 "parser.y"
{ yymsp[-1].minor.yy0 = yymsp[0].minor.yy0; }
#line 1444 "parser.c"
}
        break;
      case 48: /* unary_expression ::= MINUS unary_expression */
{  yy_destructor(yypParser,6,&yymsp[-1].minor);
#line 405 "parser.y"
{ 
      Exp ue = yymsp[0].minor.yy0;
      ue->coef *= -1;
      yymsp[-1].minor.yy0 = ue; 
    }
#line 1455 "parser.c"
}
        break;
      case 51: /* primary_expression ::= LPAREN expression RPAREN */
{  yy_destructor(yypParser,22,&yymsp[-2].minor);
#line 414 "parser.y"
{ yymsp[-2].minor.yy0 = make_Parenth(tce_tokPos,yymsp[-1].minor.yy0); }
#line 1462 "parser.c"
  yy_destructor(yypParser,23,&yymsp[0].minor);
}
        break;
      case 52: /* permut_symmetry_opt ::= */
#line 418 "parser.y"
{ yymsp[1].minor.yy0=NULL; }
#line 1469 "parser.c"
        break;
      case 53: /* permut_symmetry_opt ::= COLON symmetry_group_list */
{  yy_destructor(yypParser,21,&yymsp[-1].minor);
#line 419 "parser.y"
{ yymsp[-1].minor.yy0=NULL; }
#line 1475 "parser.c"
}
        break;
      case 54: /* permut_symmetry_opt ::= COLON ID */
{  yy_destructor(yypParser,21,&yymsp[-1].minor);
#line 421 "parser.y"
{ yymsp[-1].minor.yy0=yymsp[0].minor.yy0; }
#line 1482 "parser.c"
}
        break;
      case 57: /* symmetry_group ::= LPAREN num_list RPAREN */
{  yy_destructor(yypParser,22,&yymsp[-2].minor);
#line 426 "parser.y"
{
}
#line 1490 "parser.c"
  yy_destructor(yypParser,23,&yymsp[0].minor);
}
        break;
      case 58: /* expansion_declaration ::= EXPAND id_list SEMI */
{  yy_destructor(yypParser,24,&yymsp[-2].minor);
#line 428 "parser.y"
{
}
#line 1499 "parser.c"
  yy_destructor(yypParser,16,&yymsp[0].minor);
}
        break;
      case 59: /* volatile_declaration ::= VOLATILE id_list SEMI */
{  yy_destructor(yypParser,25,&yymsp[-2].minor);
#line 431 "parser.y"
{
}
#line 1508 "parser.c"
  yy_destructor(yypParser,16,&yymsp[0].minor);
}
        break;
      case 60: /* iteration_declaration ::= ITERATION EQUALS numerical_constant SEMI */
{  yy_destructor(yypParser,26,&yymsp[-3].minor);
#line 434 "parser.y"
{
}
#line 1517 "parser.c"
  yy_destructor(yypParser,1,&yymsp[-2].minor);
  yy_destructor(yypParser,16,&yymsp[0].minor);
}
        break;
      default:
      /* (55) symmetry_group_list ::= symmetry_group (OPTIMIZED OUT) */ assert(yyruleno!=55);
      /* (56) symmetry_group_list ::= symmetry_group_list symmetry_group */ yytestcase(yyruleno==56);
        break;
/********** End reduce actions ************************************************/
  };
  assert( yyruleno<sizeof(yyRuleInfo)/sizeof(yyRuleInfo[0]) );
  yygoto = yyRuleInfo[yyruleno].lhs;
  yysize = yyRuleInfo[yyruleno].nrhs;
  yyact = yy_find_reduce_action(yymsp[-yysize].stateno,(YYCODETYPE)yygoto);
  if( yyact <= YY_MAX_SHIFTREDUCE ){
    if( yyact>YY_MAX_SHIFT ){
      yyact += YY_MIN_REDUCE - YY_MIN_SHIFTREDUCE;
    }
    yymsp -= yysize-1;
    yypParser->yytos = yymsp;
    yymsp->stateno = (YYACTIONTYPE)yyact;
    yymsp->major = (YYCODETYPE)yygoto;
    yyTraceShift(yypParser, yyact);
  }else{
    assert( yyact == YY_ACCEPT_ACTION );
    yypParser->yytos -= yysize;
    yy_accept(yypParser);
  }
}

/*
** The following code executes when the parse fails
*/
#ifndef YYNOERRORRECOVERY
static void yy_parse_failed(
  yyParser *yypParser           /* The parser */
){
  ParseARG_FETCH;
#ifndef NDEBUG
  if( yyTraceFILE ){
    fprintf(yyTraceFILE,"%sFail!\n",yyTracePrompt);
  }
#endif
  while( yypParser->yytos>yypParser->yystack ) yy_pop_parser_stack(yypParser);
  /* Here code is inserted which will be executed whenever the
  ** parser fails */
/************ Begin %parse_failure code ***************************************/
#line 24 "parser.y"

      fprintf(stderr,"Giving up.  Parser is lost...\n");
    
#line 1569 "parser.c"
/************ End %parse_failure code *****************************************/
  ParseARG_STORE; /* Suppress warning about unused %extra_argument variable */
}
#endif /* YYNOERRORRECOVERY */

/*
** The following code executes when a syntax error first occurs.
*/
static void yy_syntax_error(
  yyParser *yypParser,           /* The parser */
  int yymajor,                   /* The major type of the error token */
  ParseTOKENTYPE yyminor         /* The minor type of the error token */
){
  ParseARG_FETCH;
#define TOKEN yyminor
/************ Begin %syntax_error code ****************************************/
#line 18 "parser.y"

      yyerror("Syntax error");
      //exit(1);
    
#line 1591 "parser.c"
/************ End %syntax_error code ******************************************/
  ParseARG_STORE; /* Suppress warning about unused %extra_argument variable */
}

/*
** The following is executed when the parser accepts
*/
static void yy_accept(
  yyParser *yypParser           /* The parser */
){
  ParseARG_FETCH;
#ifndef NDEBUG
  if( yyTraceFILE ){
    fprintf(yyTraceFILE,"%sAccept!\n",yyTracePrompt);
  }
#endif
  assert( yypParser->yytos==yypParser->yystack );
  /* Here code is inserted which will be executed whenever the
  ** parser accepts */
/*********** Begin %parse_accept code *****************************************/
/*********** End %parse_accept code *******************************************/
  ParseARG_STORE; /* Suppress warning about unused %extra_argument variable */
}

/* The main parser program.
** The first argument is a pointer to a structure obtained from
** "ParseAlloc" which describes the current state of the parser.
** The second argument is the major token number.  The third is
** the minor token.  The fourth optional argument is whatever the
** user wants (and specified in the grammar) and is available for
** use by the action routines.
**
** Inputs:
** <ul>
** <li> A pointer to the parser (an opaque structure.)
** <li> The major token number.
** <li> The minor token number.
** <li> An option argument of a grammar-specified type.
** </ul>
**
** Outputs:
** None.
*/
void Parse(
  void *yyp,                   /* The parser */
  int yymajor,                 /* The major token code number */
  ParseTOKENTYPE yyminor       /* The value for the token */
  ParseARG_PDECL               /* Optional %extra_argument parameter */
){
  YYMINORTYPE yyminorunion;
  unsigned int yyact;   /* The parser action. */
#if !defined(YYERRORSYMBOL) && !defined(YYNOERRORRECOVERY)
  int yyendofinput;     /* True if we are at the end of input */
#endif
#ifdef YYERRORSYMBOL
  int yyerrorhit = 0;   /* True if yymajor has invoked an error */
#endif
  yyParser *yypParser;  /* The parser */

  yypParser = (yyParser*)yyp;
  assert( yypParser->yytos!=0 );
#if !defined(YYERRORSYMBOL) && !defined(YYNOERRORRECOVERY)
  yyendofinput = (yymajor==0);
#endif
  ParseARG_STORE;

#ifndef NDEBUG
  if( yyTraceFILE ){
    fprintf(yyTraceFILE,"%sInput '%s'\n",yyTracePrompt,yyTokenName[yymajor]);
  }
#endif

  do{
    yyact = yy_find_shift_action(yypParser,(YYCODETYPE)yymajor);
    if( yyact <= YY_MAX_SHIFTREDUCE ){
      yy_shift(yypParser,yyact,yymajor,yyminor);
#ifndef YYNOERRORRECOVERY
      yypParser->yyerrcnt--;
#endif
      yymajor = YYNOCODE;
    }else if( yyact <= YY_MAX_REDUCE ){
      yy_reduce(yypParser,yyact-YY_MIN_REDUCE);
    }else{
      assert( yyact == YY_ERROR_ACTION );
      yyminorunion.yy0 = yyminor;
#ifdef YYERRORSYMBOL
      int yymx;
#endif
#ifndef NDEBUG
      if( yyTraceFILE ){
        fprintf(yyTraceFILE,"%sSyntax Error!\n",yyTracePrompt);
      }
#endif
#ifdef YYERRORSYMBOL
      /* A syntax error has occurred.
      ** The response to an error depends upon whether or not the
      ** grammar defines an error token "ERROR".  
      **
      ** This is what we do if the grammar does define ERROR:
      **
      **  * Call the %syntax_error function.
      **
      **  * Begin popping the stack until we enter a state where
      **    it is legal to shift the error symbol, then shift
      **    the error symbol.
      **
      **  * Set the error count to three.
      **
      **  * Begin accepting and shifting new tokens.  No new error
      **    processing will occur until three tokens have been
      **    shifted successfully.
      **
      */
      if( yypParser->yyerrcnt<0 ){
        yy_syntax_error(yypParser,yymajor,yyminor);
      }
      yymx = yypParser->yytos->major;
      if( yymx==YYERRORSYMBOL || yyerrorhit ){
#ifndef NDEBUG
        if( yyTraceFILE ){
          fprintf(yyTraceFILE,"%sDiscard input token %s\n",
             yyTracePrompt,yyTokenName[yymajor]);
        }
#endif
        yy_destructor(yypParser, (YYCODETYPE)yymajor, &yyminorunion);
        yymajor = YYNOCODE;
      }else{
        while( yypParser->yytos >= &yypParser->yystack
            && yymx != YYERRORSYMBOL
            && (yyact = yy_find_reduce_action(
                        yypParser->yytos->stateno,
                        YYERRORSYMBOL)) >= YY_MIN_REDUCE
        ){
          yy_pop_parser_stack(yypParser);
        }
        if( yypParser->yytos < yypParser->yystack || yymajor==0 ){
          yy_destructor(yypParser,(YYCODETYPE)yymajor,&yyminorunion);
          yy_parse_failed(yypParser);
#ifndef YYNOERRORRECOVERY
          yypParser->yyerrcnt = -1;
#endif
          yymajor = YYNOCODE;
        }else if( yymx!=YYERRORSYMBOL ){
          yy_shift(yypParser,yyact,YYERRORSYMBOL,yyminor);
        }
      }
      yypParser->yyerrcnt = 3;
      yyerrorhit = 1;
#elif defined(YYNOERRORRECOVERY)
      /* If the YYNOERRORRECOVERY macro is defined, then do not attempt to
      ** do any kind of error recovery.  Instead, simply invoke the syntax
      ** error routine and continue going as if nothing had happened.
      **
      ** Applications can set this macro (for example inside %include) if
      ** they intend to abandon the parse upon the first syntax error seen.
      */
      yy_syntax_error(yypParser,yymajor, yyminor);
#ifndef YYNOERRORRECOVERY
      yypParser->yyerrcnt = -1;
#endif
      yy_destructor(yypParser,(YYCODETYPE)yymajor,&yyminorunion);
      yymajor = YYNOCODE;
      
#else  /* YYERRORSYMBOL is not defined */
      /* This is what we do if the grammar does not define ERROR:
      **
      **  * Report an error message, and throw away the input token.
      **
      **  * If the input token is $, then fail the parse.
      **
      ** As before, subsequent error messages are suppressed until
      ** three input tokens have been successfully shifted.
      */
      if( yypParser->yyerrcnt<=0 ){
        yy_syntax_error(yypParser,yymajor, yyminor);
      }
      yypParser->yyerrcnt = 3;
      yy_destructor(yypParser,(YYCODETYPE)yymajor,&yyminorunion);
      if( yyendofinput ){
        yy_parse_failed(yypParser);
      }
      yymajor = YYNOCODE;
#endif
    }
  }while( yymajor!=YYNOCODE && yypParser->yytos>yypParser->yystack );
#ifndef NDEBUG
  if( yyTraceFILE ){
    yyStackEntry *i;
    char cDiv = '[';
    fprintf(yyTraceFILE,"%sReturn. Stack=",yyTracePrompt);
    for(i=&yypParser->yystack[1]; i<=yypParser->yytos; i++){
      fprintf(yyTraceFILE,"%c%s", cDiv, yyTokenName[i->major]);
      cDiv = ' ';
    }
    fprintf(yyTraceFILE,"]\n");
  }
#endif
  return;
}


import ply
import ply.lex as lex
import ply.yacc as yacc
from error import InputError, FrontEndError
from absyn import (
    TranslationUnit,
    CompoundElem,
    RangeDecl,
    IndexDecl,
    ArrayDecl,
    ExpandDecl,
    IterationDecl,
    VolatileDecl,
    AssignStmt,
    Identifier,
    Parenth,
    NumConst,
    Array,
    Addition,
    Multiplication
)


# Reserved words
reserved = ['RANGE', 'INDEX', 'ARRAY', 'EXPAND', 'VOLATILE', 'ITERATION']

tokens = reserved + [
    # Literals (identifier, integer constant, float constant)
    'ID', 'ICONST', 'FCONST',

    # Operators (+,-,*)
    'PLUS', 'MINUS', 'TIMES',

    # Assignment (=, *=, +=, -=)
    'EQUALS', 'TIMESEQUAL', 'PLUSEQUAL', 'MINUSEQUAL',

    # Delimeters ( ) [ ] { } < > , ;
    'LPAREN', 'RPAREN',
    'LBRACKET', 'RBRACKET',
    'LBRACE', 'RBRACE',
    'LARROW', 'RARROW',
    'COMMA', 'SEMI', 'COLON',
    ]

# Operators
t_PLUS             = r'\+'
t_MINUS            = r'-'
t_TIMES            = r'\*'

# Assignment operators
t_EQUALS           = r'='
t_TIMESEQUAL       = r'\*='
t_PLUSEQUAL        = r'\+='
t_MINUSEQUAL       = r'-='

# Delimeters
t_LPAREN           = r'\('
t_RPAREN           = r'\)'
t_LBRACKET         = r'\['
t_RBRACKET         = r'\]'
t_LBRACE           = r'\{'
t_RBRACE           = r'\}'
t_LARROW           = r'\<'
t_RARROW           = r'\>'
t_COMMA            = r','
t_SEMI             = r';'
t_COLON            = r':'

# Ignored characters
t_ignore = ' \t'

# Identifiers and reserved words
reserved_map = {}
for r in reserved:
    reserved_map[r.lower()] = r


def t_ID(t):
    r'[A-Za-z_]\w*'
    t.type = reserved_map.get(t.value, 'ID')
    return t

# Integer literal
t_ICONST = r'\d+'

# Floating literal
t_FCONST = r'((\d+\.\d*)|(\.\d+))'


# Newlines
def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += t.value.count('\n')


# Comments
def t_comment(t):
    r"\043.*"  # \043 is '#'
    pass


# Syntactical error
def t_error(t):
    raise InputError('syntactical error: "%s"' % t.value[0], t.lineno)


# translation-unit
def p_translation_unit(p):
    'translation_unit : compound_element_list'
    p[0] = TranslationUnit(p[1])


# compound-element-list
def p_compound_element_list_1(p):
    'compound_element_list :'
    p[0] = []


def p_compound_element_list_2(p):
    'compound_element_list : compound_element'
    p[0] = [p[1]]


def p_compound_element_list_3(p):
    'compound_element_list : compound_element_list compound_element'
    p[1].append(p[2])
    p[0] = p[1]


# compound-element
def p_compound_element(p):
    'compound_element : LBRACE element_list RBRACE'
    p[0] = CompoundElem(p[2])


# element-list
def p_element_list_1(p):
    'element_list :'
    p[0] = []


def p_element_list_2(p):
    'element_list : element'
    p[0] = p[1]


def p_element_list_3(p):
    'element_list : element_list element'
    p[1].extend(p[2])
    p[0] = p[1]


# element
def p_element_1(p):
    'element : declaration'
    p[0] = p[1]


def p_element_2(p):
    'element : statement'
    p[0] = p[1]


# declaration
def p_declaration_1(p):
    'declaration : range_declaration'
    p[0] = p[1]


def p_declaration_2(p):
    'declaration : index_declaration'
    p[0] = p[1]


def p_declaration_3(p):
    'declaration : array_declaration'
    p[0] = p[1]


def p_declaration_4(p):
    'declaration : expansion_declaration'
    p[0] = p[1]


def p_declaration_5(p):
    'declaration : volatile_declaration'
    p[0] = p[1]


def p_declaration_6(p):
    'declaration : iteration_declaration'
    p[0] = p[1]


# range-declaration
def p_range_declaration(p):
    'range_declaration : RANGE id_list EQUALS numerical_constant SEMI'
    rdecls = []
    for id in p[2]:
        range_value = p[4].replicate()
        range_value.line_no = p.lineno(4)
        rdecl = RangeDecl(id.name, range_value)
        rdecl.line_no = id.line_no
        rdecls.append(rdecl)
    p[0] = rdecls


# index-declaration
def p_index_declaration(p):
    'index_declaration : INDEX id_list EQUALS identifier SEMI'
    idecls = []
    for id in p[2]:
        range_id = p[4].replicate()
        range_id.line_no = p.lineno(4)
        idecl = IndexDecl(id.name, range_id)
        idecl.line_no = id.line_no
        idecls.append(idecl)
    p[0] = idecls


# array-declaration
def p_array_declaration(p):
    'array_declaration : ARRAY array_structure_list SEMI'
    p[0] = p[2]


# array-structure-list
def p_array_structure_list_1(p):
    'array_structure_list : array_structure'
    p[0] = [p[1]]


def p_array_structure_list_2(p):
    'array_structure_list : array_structure_list COMMA array_structure'
    p[1].append(p[3])
    p[0] = p[1]


# array-structure
def p_array_structure(p):
    'array_structure : ID LPAREN LBRACKET id_list_opt RBRACKET LBRACKET id_list_opt RBRACKET permut_symmetry_opt vertex_symmetry_opt RPAREN'
    adecl = ArrayDecl(p[1], p[4], p[7], p[9], p[10])
    adecl.line_no = p.lineno(1)
    p[0] = adecl


# permutation-symmetry
def p_permut_symmetry_opt_1(p):
    'permut_symmetry_opt :'
    p[0] = []


def p_permut_symmetry_opt_2(p):
    'permut_symmetry_opt : COLON symmetry_group_list'
    p[0] = p[2]


def p_symmetry_group_list_1(p):
    'symmetry_group_list : symmetry_group'
    p[0] = [p[1]]


def p_symmetry_group_list_2(p):
    'symmetry_group_list : symmetry_group_list symmetry_group'
    p[1].append(p[2])
    p[0] = p[1]


def p_symmetry_group(p):
    'symmetry_group : LPAREN num_list RPAREN'
    p[0] = p[2]


# vertex-symmetry
def p_vertex_symmetry_opt_1(p):
    'vertex_symmetry_opt :'
    p[0] = []


def p_vertex_symmetry_opt_2(p):
    'vertex_symmetry_opt : COLON LARROW num_list RARROW LARROW num_list RARROW'
    p[0] = [p[3],p[6]]


# expansion-declaration
def p_expansion_declaration(p):
    'expansion_declaration : EXPAND id_list SEMI'
    edecls = []
    for id in p[2]:
        edecl = ExpandDecl(id)
        edecl.line_no = id.line_no
        edecls.append(edecl)
    p[0] = edecls


# volatile-declaration
def p_volatile_declaration(p):
    'volatile_declaration : VOLATILE id_list SEMI'
    edecls = []
    for id in p[2]:
        edecl = VolatileDecl(id)
        edecl.line_no = id.line_no
        edecls.append(edecl)
    p[0] = edecls


# iteration-declaration
def p_iteration_declaration(p):
    'iteration_declaration : ITERATION EQUALS numerical_constant SEMI'
    idecl = IterationDecl(p[3])
    idecl.line_no = p.lineno(1)
    p[0] = [idecl]


# statement
def p_statement(p):
    'statement : assignment_statement'
    p[0] = [p[1]]


# assignment-statement
def p_assignment_statement(p):
    'assignment_statement : expression assignment_operator expression SEMI'
    if (p[2] == '='):
        astmt = AssignStmt(p[1], p[3])
    elif (p[2] == '*='):
        lhs = p[1].replicate()
        lhs.line_no = p[1].line_no
        rhs = Parenth(p[3])
        mult = Multiplication([lhs, rhs])
        mult.line_no = p.lineno(2)
        astmt = AssignStmt(p[1], mult)
    elif (p[2] == '+='):
        lhs = p[1].replicate()
        lhs.line_no = p[1].line_no
        rhs = Parenth(p[3])
        add = Addition([lhs, rhs])
        add.line_no = p.lineno(2)
        astmt = AssignStmt(p[1], add)
    elif (p[2] == '-='):
        lhs = p[1].replicate()
        lhs.line_no = p[1].line_no
        rhs = Parenth(p[3])
        rhs.inverseSign()
        add = Addition([lhs, rhs])
        add.line_no = p.lineno(2)
        astmt = AssignStmt(p[1], add)
    else:
        raise FrontEndError('%s: unknown assignment operator' % __name__)
    astmt.line_no = p.lineno(2)
    p[0] = astmt


# assignment_operator:
def p_assignment_operator(p):
    '''assignment_operator : EQUALS
                           | TIMESEQUAL
                           | PLUSEQUAL
                           | MINUSEQUAL
                           '''
    p[0] = p[1]


# expression
def p_expression(p):
    'expression : additive_expression'
    p[0] = p[1]


# additive-expression
def p_additive_expression_1(p):
    'additive_expression : multiplicative_expression'
    p[0] = p[1]


def p_additive_expression_2(p):
    'additive_expression : additive_expression PLUS multiplicative_expression'
    subexps = []
    line_no = p.lineno(2)
    if (isinstance(p[1], Addition)):
        subexps.extend(p[1].subexps)
        line_no = p[1].line_no
    else:
        subexps.append(p[1])
    if (isinstance(p[3], Addition)):
        subexps.extend(p[3].subexps)
    else:
        subexps.append(p[3])
    aexp = Addition(subexps)
    aexp.line_no = line_no
    p[0] = aexp


def p_additive_expression_3(p):
    'additive_expression : additive_expression MINUS multiplicative_expression'
    p[3].inverseSign()
    subexps = []
    line_no = p.lineno(2)
    if (isinstance(p[1], Addition)):
        subexps.extend(p[1].subexps)
        line_no = p[1].line_no
    else:
        subexps.append(p[1])
    if (isinstance(p[3], Addition)):
        subexps.extend(p[3].subexps)
    else:
        subexps.append(p[3])
    aexp = Addition(subexps)
    aexp.line_no = line_no
    p[0] = aexp


# multiplicative-expression
def p_multiplicative_expression_1(p):
    'multiplicative_expression : unary_expression'
    p[0] = p[1]


def p_multiplicative_expression_2(p):
    'multiplicative_expression : multiplicative_expression TIMES unary_expression'
    coef = 1
    subexps = []
    line_no = p.lineno(2)
    if (isinstance(p[1], Multiplication)):
        coef *= p[1].coef
        subexps.extend(p[1].subexps)
        line_no = p[1].line_no
    else:
        subexps.append(p[1])
    if (isinstance(p[3], Multiplication)):
        coef *= p[3].coef
        subexps.extend(p[3].subexps)
    else:
        subexps.append(p[3])
    mexp = Multiplication(subexps, coef)
    mexp.line_no = line_no
    p[0] = mexp


# unary-expression
def p_unary_expression_1(p):
    'unary_expression : primary_expression'
    p[0] = p[1]


def p_unary_expression_2(p):
    'unary_expression : PLUS unary_expression'
    p[0] = p[2]


def p_unary_expression_3(p):
    'unary_expression : MINUS unary_expression'
    p[2].inverseSign()
    p[0] = p[2]


# primary-expression
def p_primary_expression_1(p):
    'primary_expression : numerical_constant'
    p[0] = p[1]


def p_primary_expression_2(p):
    'primary_expression : array_reference'
    p[0] = p[1]


def p_primary_expression_3(p):
    'primary_expression : LPAREN expression RPAREN'
    p[0] = Parenth(p[2])


# numerical-constant
def p_numerical_constant_1(p):
    'numerical_constant : ICONST'
    if (len(p[1]) > 1 and p[1].startswith('0')):
        raise InputError('invalid decimal integer: "%s"' % p[1], p.lineno(1))
    nconst = NumConst(float(p[1]))
    nconst.line_no = p.lineno(1)
    p[0] = nconst


def p_numerical_constant_2(p):
    'numerical_constant : FCONST'
    if (p[1].startswith('0') and p[1][1] != '.'):
        raise InputError('invalid decimal float: "%s"' % p[1], p.lineno(1))
    nconst = NumConst(float(p[1]))
    nconst.line_no = p.lineno(1)
    p[0] = nconst


# array-reference
def p_array_reference_1(p):
    'array_reference : ID'
    aref = Array(p[1], [])
    aref.line_no = p.lineno(1)
    p[0] = aref


def p_array_reference_2(p):
    'array_reference : ID LBRACKET id_list RBRACKET'
    aref = Array(p[1], p[3])
    aref.line_no = p.lineno(1)
    p[0] = aref


# id-list
def p_id_list_opt_1(p):
    'id_list_opt :'
    p[0] = []


def p_id_list_opt_2(p):
    'id_list_opt : id_list'
    p[0] = p[1]


def p_id_list_1(p):
    'id_list : identifier'
    p[0] = [p[1]]


def p_id_list_2(p):
    'id_list : id_list COMMA identifier'
    p[1].append(p[3])
    p[0] = p[1]


# num-list
def p_num_list_1(p):
    'num_list : numerical_constant'
    p[0] = [p[1]]


def p_num_list_2(p):
    'num_list : num_list COMMA numerical_constant'
    p[1].append(p[3])
    p[0] = p[1]


# identifier
def p_identifier(p):
    'identifier : ID'
    id = Identifier(p[1])
    id.line_no = p.lineno(1)
    p[0] = id


# Grammatical error
def p_error(p):
    raise InputError('grammatical error: "%s"' % p.value, p.lineno)


def createParser():
    lex.lex()
    parser = yacc.yacc(method='LALR', debug=0)
    return parser

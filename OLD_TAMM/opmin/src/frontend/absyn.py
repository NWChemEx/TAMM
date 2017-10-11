#
# Absyn
#  |
#  +-- TranslationUnit
#  |
#  +-- CompoundElem
#  |
#  +-- Elem
#  |    |
#  |    +-- Decl
#  |    |    |
#  |    |    +-- RangeDecl
#  |    |    +-- IndexDecl
#  |    |    +-- ArrayDecl
#  |    |    +-- ExpandDecl
#  |    |    +-- VolatileDecl
#  |    |    +-- IterationDecl
#  |    |
#  |    +-- Stmt
#  |         |
#  |         +-- AssignStmt
#  |
#  +-- Identifier
#  |
#  +-- Exp
#       |
#       +-- Parenth
#       +-- NumConst
#       +-- Array
#       +-- Multiplication
#       +-- Addition
#

#-----------------------------------------

class Absyn:
    def __init__(self):
        pass

    def replicate(self):
        raise NotImplementedError('%s: abstract function "replicate" unimplemented' % (self.__class__.__name__))
    
    def __repr__(self):
        raise NotImplementedError('%s: abstract function "__repr__" unimplemented' % (self.__class__.__name__))

    def __str__(self):
        return repr(self)

#-----------------------------------------

class TranslationUnit(Absyn):
    def __init__(self, comp_elems):
        Absyn.__init__(self)
        self.comp_elems = comp_elems

    def replicate(self):
        r_comp_elems = [ce.replicate() for ce in self.comp_elems]
        return TranslationUnit(r_comp_elems)

    def __repr__(self):
        return '\n'.join(map(lambda x: '\n' + str(x), self.comp_elems))

#-----------------------------------------

class CompoundElem(Absyn):
    def __init__(self, elems):
        Absyn.__init__(self)
        self.elems = elems

    def replicate(self):
        r_elems = [e.replicate() for e in self.elems]
        return CompoundElem(r_elems)

    def __repr__(self):
        s = '{'
        s += ''.join(map(lambda x: '\n  ' + str(x), self.elems))
        if (len(self.elems) > 0):
            s += '\n'
        s += '}'
        return s

#-----------------------------------------

class Elem(Absyn):
    def __init__(self):
        Absyn.__init__(self)

#-----------------------------------------

class Decl(Elem):
    def __init__(self):
        Elem.__init__(self)

class RangeDecl(Decl):
    def __init__(self, name, value):
        Decl.__init__(self)
        self.name = name
        self.value = value

    def replicate(self):
        return RangeDecl(self.name, self.value.replicate())

    def __repr__(self):
        s = ''
        s += 'range ' + str(self.name) + ' = ' + str(self.value) + ';'
        return s

class IndexDecl(Decl):
    def __init__(self, name, range):
        Decl.__init__(self)
        self.name = name
        self.range = range

    def replicate(self):
        return IndexDecl(self.name, self.range.replicate())

    def __repr__(self):
        s = ''
        s += 'index ' + str(self.name) + ' = ' + str(self.range) + ';'
        return s    

class ArrayDecl(Decl):
    def __init__(self, name, upper_ranges, lower_ranges, sym_groups, vsym_groups):
        Decl.__init__(self)
        self.name = name
        self.upper_ranges = upper_ranges
        self.lower_ranges = lower_ranges
        self.sym_groups = sym_groups
        self.vsym_groups = vsym_groups

    def replicate(self):
        r_uranges = [r.replicate() for r in self.upper_ranges]
        r_lranges = [r.replicate() for r in self.lower_ranges]
        r_sgroups = [[i.replicate() for i in g] for g in self.sym_groups]
        r_vsgroups = [[i.replicate() for i in g] for g in self.vsym_groups]
        return ArrayDecl(self.name, r_uranges, r_lranges, r_sgroups, r_vsgroups)

    def __repr__(self):
        s = ''
        s += 'array ' + str(self.name) + '(['
        s += ','.join(map(str, self.upper_ranges))
        s += ']['
        s += ','.join(map(str, self.lower_ranges))        
        s += ']:'
        for g in self.sym_groups:
            s += '(' + ','.join(map(str, g)) + ')'
        s += ':'
        for g in self.vsym_groups:
            s += '<' + ','.join(map(str, g)) + '>'
        s += ');'
        return s

class ExpandDecl(Decl):
    def __init__(self, arr):
        Decl.__init__(self)
        self.arr = arr

    def replicate(self):
        return ExpandDecl(self.arr.replicate())

    def __repr__(self):
        s = ''
        s += 'expand ' + str(self.arr) + ';'
        return s
        
class VolatileDecl(Decl):
    def __init__(self, arr):
        Decl.__init__(self)
        self.arr = arr

    def replicate(self):
        return VolatileDecl(self.arr.replicate())

    def __repr__(self):
        s = ''
        s += 'volatile ' + str(self.arr) + ';'
        return s

class IterationDecl(Decl):
    def __init__(self, value):
        Decl.__init__(self)
        self.value = value

    def replicate(self):
        return IterationDecl(self.value.replicate())

    def __repr__(self):
        s = ''
        s += 'iteration = ' + str(self.value) + ';'
        return s

#-----------------------------------------

class Stmt(Elem):
    def __init__(self):
        Elem.__init__(self)

class AssignStmt(Stmt):
    def __init__(self, lhs, rhs):
        Stmt.__init__(self)
        self.lhs = lhs
        self.rhs = rhs
        
    def replicate(self):
        return AssignStmt(self.lhs.replicate(), self.rhs.replicate())

    def __repr__(self):
        s = ''
        s += str(self.lhs)
        s += ' = '
        s += str(self.rhs)
        s += ';'
        return s

#-----------------------------------------

class Identifier(Absyn):
    def __init__(self, name):
        Absyn.__init__(self)
        self.name = name

    def replicate(self):
        return Identifier(self.name)

    def __repr__(self):
        return str(self.name)

#-----------------------------------------

class Exp(Absyn):
    def __init__(self, coef):
        Absyn.__init__(self)
        self.coef = coef

    def isNegative(self):
        return self.coef < 0

    def isPositive(self):
        return self.coef > 0

    def setCoef(self, coef):
        raise NotImplementedError('%s: abstract function "setCoef" unimplemented' % (self.__class__.__name__))
    
    def multiplyCoef(self, coef):
        raise NotImplementedError('%s: abstract function "multiplyCoef" unimplemented' % (self.__class__.__name__))
    
    def inverseSign(self):
        raise NotImplementedError('%s: abstract function "inverseSign" unimplemented' % (self.__class__.__name__))

class Parenth(Exp):
    def __init__(self, exp, coef = 1):
        Exp.__init__(self, coef)
        self.exp = exp
        
    def replicate(self):
        r_exp = self.exp.replicate()
        return Parenth(r_exp, self.coef)

    def __repr__(self):
        s = ''
        if (self.coef == -1):
            s += '-'
        elif (self.coef != -1 and self.coef != 1):
            s += '(' + str(self.coef) + ' * '
        s += '('
        s += str(self.exp)
        s += ')'
        if (self.coef != -1 and self.coef != 1):
            s += ')'
        return s

    def setCoef(self, coef):
        self.exp.setCoef(coef)
    
    def multiplyCoef(self, coef):
        self.exp.multiplyCoef(coef)
    
    def inverseSign(self):
        self.exp.inverseSign()
    
class NumConst(Exp):
    def __init__(self, value, coef = 1):
        Exp.__init__(self, coef)
        self.value = value

    def replicate(self):
        return NumConst(self.value, self.coef)

    def __repr__(self):
        s = ''
        if (self.coef == -1):
            s += '-'
        elif (self.coef != -1 and self.coef != 1):
            s += '(' + str(self.coef) + ' * '
        s += str(self.value)
        if (self.coef != -1 and self.coef != 1):
            s += ')'
        return s

    def setCoef(self, coef):
        self.coef = coef
    
    def multiplyCoef(self, coef):
        self.coef *= coef
    
    def inverseSign(self):
        self.multiplyCoef(-1)

class Array(Exp):
    def __init__(self, name, inds, coef = 1):
        Exp.__init__(self, coef)
        self.name = name
        self.inds = inds

    def replicate(self):
        r_inds = [i.replicate() for i in self.inds]
        return Array(self.name, r_inds, self.coef)

    def __repr__(self):
        s = ''
        if (self.coef == -1):
            s += '-'
        elif (self.coef != -1 and self.coef != 1):
            s += '(' + str(self.coef) + ' * '
        s += str(self.name)
        if (len(self.inds) > 0):
            s += '['
            s += ','.join(map(lambda x: str(x.name), self.inds))
            s += ']'
        if (self.coef != -1 and self.coef != 1):
            s += ')'
        return s

    def setCoef(self, coef):
        self.coef = coef
    
    def multiplyCoef(self, coef):
        self.coef *= coef
    
    def inverseSign(self):
        self.multiplyCoef(-1)

class Addition(Exp):
    def __init__(self, subexps, coef = 1):
        Exp.__init__(self, coef)
        self.subexps = subexps

    def replicate(self):
        r_subexps = [e.replicate() for e in self.subexps]
        return Addition(r_subexps, self.coef)
    
    def __repr__(self):
        s = ''
        if (self.coef == -1):
            s += '-'
        elif (self.coef != -1 and self.coef != 1):
            s += '(' + str(self.coef) + ' * '
        s += '('
        s += ' + '.join(map(str, self.subexps))
        s += ')'
        if (self.coef != -1 and self.coef != 1):
            s += ')'
        return s

    def setCoef(self, coef):
        for e in self.subexps:
            e.setCoef(coef)

    def multiplyCoef(self, coef):
        for e in self.subexps:
            e.multiplyCoef(coef)

    def inverseSign(self):
        for e in self.subexps:
            e.inverseSign()
            
class Multiplication(Exp):
    def __init__(self, subexps, coef = 1):
        Exp.__init__(self, coef)
        self.subexps = subexps

    def replicate(self):
        r_subexps = [e.replicate() for e in self.subexps]
        return Multiplication(r_subexps, self.coef)

    def __repr__(self):
        s = ''
        if (self.coef == -1):
            s += '-'
        s += '('
        if (self.coef != -1 and self.coef != 1):
            s += str(self.coef) + ' * '
        s += ' * '.join(map(str, self.subexps))
        s += ')'
        return s

    def setCoef(self, coef):
        self.coef = coef
    
    def multiplyCoef(self, coef):
        self.coef *= coef
    
    def inverseSign(self):
        self.multiplyCoef(-1)


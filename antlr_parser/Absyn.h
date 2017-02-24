//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------

// TAMM Parser Class Heirarchy
// Absyn
//  |
//  +-- CompilationUnit
//  |
//  +-- Element
//  |    |
//  |    +-- DeclarationList
//  |    +-- Declaration
//  |    |    |
//  |    |    +-- RangeDeclaration
//  |    |    +-- IndexDeclaration
//  |    |    +-- ArrayDeclaration
//  |    |
//  |    +-- Statement
//  |         |
//  |         +-- AssignStatement
//  |
//  +-- Identifier
//  |
//  +-- Expression
//       |
//       +-- Parenth
//       +-- NumConst
//       +-- Array
//       +-- Multiplication
//       +-- Addition
//
//-----------------------------------------

#ifndef __TAMM_ABSYN_H_
#define __TAMM_ABSYN_H_

#include "SymbolTable.h"
#include <type_traits>


namespace tamm {

/* Forward Declarations */
class Element;
class Identifier;
class Declaration;
class Expression;
class DeclarationList;


/* The Absyn Hierarchy */

class Absyn //Root of the AST
{
    public: 

        const int line;     /// Line number
        const int position; /// column number in line

        Absyn(const int line, const int position): line(line), position(position) {}

        enum kAbsyn {
            kCompilationUnit, kElement, kIdentifier, kExpression
        };

        /** Implemented in all direct subclasses.
         * Used to identify a direct subclass of Declaration.
         * Returns the enum value for the subclass
         * that calls this method.
         */
        virtual int getAbsynType() = 0;

        virtual ~Absyn() {}

};


class Element : public Absyn {
    public:
        /// An enum to identify direct subclasses of Element.
        enum kElement {
            kDeclarationList,
            kDeclaration,
            kStatement
        };

        Element(const int line, const int position): Absyn(line, position) {}

        virtual int getElementType() = 0;

        int getAbsynType() {
            return Absyn::kElement;
        }

        virtual ~Element() { }
};

class Declaration : public Element {
    public:
        /// An enum to identify direct subclasses of Declaration.
        enum kDeclaration {
            kRangeDeclaration,
            kIndexDeclaration,
            kArrayDeclaration,
            kExpandDeclaration,
            kVolatileDeclaration,
            kIterationDeclaration
        };

        Declaration(const int line, const int position): Element(line, position) {}

        virtual int getDeclType() = 0;

        int getElementType() {
            return Element::kDeclaration;
        }

        virtual ~Declaration() { }
};


class ArrayDeclaration : public Declaration {
    public:
          const std::string name;
          const std::vector<Identifier*> upper_indices;
          const std::vector<Identifier*> lower_indices;
          std::string irrep;

          ArrayDeclaration(const int line, const int position, const std::string name,
                           const std::vector<Identifier*>& upper_indices, 
                           const std::vector<Identifier*>& lower_indices)
                           : Declaration(line,position), name(name), 
                             upper_indices(upper_indices), lower_indices(lower_indices) {}

          int getDeclType() {
              return Declaration::kArrayDeclaration;
          }
};

class IndexDeclaration : public Declaration {
    public:
          const std::string name;
          const std::string range_id;

          IndexDeclaration(const int line, const int position, 
                           const std::string name, const std::string range_id)
            : Declaration(line,position), name(name), range_id(range_id) {}

          int getDeclType() {
              return Declaration::kIndexDeclaration;
          }
};

class RangeDeclaration : public Declaration {
    public:
          const int value;
          const std::string name;

          RangeDeclaration(const int line, const int position, const std::string name, const int value)
            : Declaration(line,position), name(name), value(value) {}

          int getDeclType() {
              return Declaration::kRangeDeclaration;
          }
};



class Statement : public Element {
    public:
        /// An enum to identify direct subclasses of Statement.
        enum kStatement {
            kAssignStatement
        };

        Statement(const int line, const int position): Element(line, position) {}

        virtual int getStatementType() = 0;

        int getElementType() {
            return kElement::kStatement;
        }

        virtual ~Statement() { }
};


class Expression : public Absyn {
public:
    enum kExpression {
        kParenth, kNumConst, kArrayRef, kAddition, kMultiplication
    };

   Expression(const int line, const int position): Absyn(line,position) {} 

    virtual int getExpressionType() = 0;

    int getAbsynType() {
        return Absyn::kExpression;
    }

    virtual ~Expression() {}
};


class Array: public Expression {
    
    public:

        const std::string name;
        const std::vector<Identifier*> indices;

        Array(const int line, const int position, 
              const std::string name, const std::vector<Identifier*>& indices) 
              : Expression(line,position), name(name), indices(indices) {} 

     int getExpressionType() { return Expression::kArrayRef; }
};

class AssignStatement: public Statement {
    public:
        const Array* const lhs;
        const Expression* const rhs;
        const std::string label;
        const std::string assign_op;

        AssignStatement(const int line, const int position, const std::string assign_op,
                        const Array* const lhs, const Expression* const rhs)
                        : Statement(line, position), assign_op(assign_op), lhs(lhs), rhs(rhs) {}

        AssignStatement(const int line, const int position, const std::string label, 
                        const std::string assign_op, const Array* const lhs, const Expression* const rhs)
                        : Statement(line,position), label(label), assign_op(assign_op), lhs(lhs), rhs(rhs) {}
        
        int getStatementType() {
            return Statement::kAssignStatement;
        }

};



class DeclarationList: public Element {
public:
    const std::vector<Declaration*> dlist;
    DeclarationList(const int line, const std::vector<Declaration*> &dlist): Element(line,0), dlist(dlist) {}

    int getElementType() {
       return Element::kDeclarationList;
    }
};


// class Parenth: public Expression {
//     public:
//         const Expression* const expression;

//         Parenth(const Expression* const expression): 
//                     Expression(0,0), expression(expression) {}

//         int getExpressionType() { return Expression::kParenth; }
// };


class NumConst: public Expression {
    public:
        const float value;

        NumConst(const int line, const int position, const float value): 
                 Expression(line,position), value(value) {}

     int getExpressionType() { return Expression::kNumConst; }
};



class Addition: public Expression {
    public:
        const bool first_op;
        const std::vector<Expression*> subexps;
        const std::vector<std::string> add_operators;

    Addition(const int line, const int position, const std::vector<Expression*>& subexps)
             : Expression(line,position), subexps(subexps), first_op(false) {}
    
    Addition(const int line, const int position, const std::vector<Expression*>& subexps,
             const std::vector<std::string> &add_operators, const bool first_op)
             : Expression(line,position), subexps(subexps), 
               add_operators(add_operators), first_op(first_op) {}
    
    int getExpressionType() { return Expression::kAddition; }

};



class Multiplication: public Expression {
    public:
        const std::vector<Expression*> subexps;

        Multiplication(const int line, const int position, const std::vector<Expression*>& subexps)
                        : Expression(line,position), subexps(subexps) {}
    
        int getExpressionType() { return Expression::kMultiplication; }
    
};


class ElementList //group of declarations and statements corresponding to a single input
{
public:
    const std::vector<Element*> elist;
    ElementList(const std::vector<Element*> &e): elist(e) {}
};


class Identifier: public Absyn {
public:

    const std::string name;
    Identifier(const int line, const int position, const std::string name)
                : Absyn(line,position), name(name) {}

    int getAbsynType() {
       return Absyn::kIdentifier;
    }
};

class IdentifierList {
public:
    std::vector<Identifier*> idlist;
    IdentifierList(const std::vector<Identifier*> &idlist): idlist(idlist) {}
};


class CompoundElement  //represents a single input enclosed in { .. }
{
public:
    const ElementList* const elist;
    CompoundElement(const ElementList* const elist): elist(elist) {}

};

class CompilationUnit : public Absyn {
public:
    const std::vector<CompoundElement*> celist;
    const SymbolTable* symbol_table;
    CompilationUnit(const std::vector<CompoundElement*>& celist, const SymbolTable* symbol_table)
                   : Absyn(0,0), celist(celist), symbol_table(symbol_table) {}

        int getAbsynType() {
            return Absyn::kCompilationUnit;
        }
};

} //namespcae tamm

#endif



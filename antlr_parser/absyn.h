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

#include "util.h"
#include <type_traits>


namespace tamm {

/* Forward Declarations */
class Element;
class Declaration;
class Expression;
class DeclarationList;


/* The Absyn Hierarchy */

class Absyn //Root of the AST
{
    public: 

        int line_number;
        int position;

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

        virtual int getDeclType() = 0;

        int getElementType() {
            return Element::kDeclaration;
        }

        virtual ~Declaration() { }
};


class ArrayDeclaration : public Declaration {
    public:
          const std::string name;
          const std::vector<std::string> upper_indices;
          const std::vector<std::string> lower_indices;
          std::string irrep;

          ArrayDeclaration(const std::string name, const std::vector<std::string>& upper_indices, const std::vector<std::string>& lower_indices)
                        : name(name), upper_indices(upper_indices), lower_indices(lower_indices)  {}

          int getDeclType() {
              return Declaration::kArrayDeclaration;
          }
};

class IndexDeclaration : public Declaration {
    public:
          const std::string name;
          const std::string range_id;

          IndexDeclaration(const std::string name, const std::string range_id): name(name), range_id(range_id) {}

          int getDeclType() {
              return Declaration::kIndexDeclaration;
          }
};

class RangeDeclaration : public Declaration {
    public:
          const int value;
          const std::string name;

          RangeDeclaration(const std::string name, const int value): name(name), value(value) {}

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

    virtual int getExpressionType() = 0;

    int getAbsynType() {
        return Absyn::kExpression;
    }

    virtual ~Expression() {}
};


class Array: public Expression {
    public:
        const std::string name;
        const std::vector<std::string> indices;

        Array(const std::string name, const std::vector<std::string>& indices): name(name), indices(indices) {}

     int getExpressionType() { return Expression::kArrayRef; }
};

class AssignStatement: public Statement {
    public:
        const Array* const lhs;
        const Expression* const rhs;
        const std::string label;
        const std::string assign_op;

        AssignStatement(const std::string assign_op, const Array* const lhs, const Expression* const rhs)
                        : assign_op(assign_op), lhs(lhs), rhs(rhs) {}

        AssignStatement(const std::string label, const std::string assign_op, const Array* const lhs, const Expression* const rhs)
                        : label(label), assign_op(assign_op), lhs(lhs), rhs(rhs) {}
        
        int getStatementType() {
            return Statement::kAssignStatement;
        }

};



class DeclarationList: public Element {
public:
    const std::vector<Declaration*> dlist;
    DeclarationList(const std::vector<Declaration*> &dlist): dlist(dlist) {}

    int getElementType() {
       return Element::kDeclarationList;
    }
};


class Parenth: public Expression {
    public:
        const Expression* const expression;

        Parenth(const Expression* const expression): expression(expression) {}

     int getExpressionType() { return Expression::kParenth; }
};


class NumConst: public Expression {
    public:
        const float value;

        NumConst(const float value): value(value) {}

     int getExpressionType() { return Expression::kNumConst; }
};



class Addition: public Expression {
    public:
        const bool first_op;
        const std::vector<Expression*> subexps;
        const std::vector<std::string> add_operators;

    Addition(const std::vector<Expression*>& subexps): subexps(subexps), first_op(false) {}
    Addition(const std::vector<Expression*>& subexps, const std::vector<std::string> &add_operators, const bool first_op)
            : subexps(subexps), add_operators(add_operators), first_op(first_op) {}
    int getExpressionType() { return Expression::kAddition; }

};



class Multiplication: public Expression {
    public:
        const std::vector<Expression*> subexps;

        Multiplication(const std::vector<Expression*>& subexps): subexps(subexps) {}
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
    Identifier(const std::string name): name(name) {}

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
    CompilationUnit(const std::vector<CompoundElement*>& celist): celist(celist) {}

        int getAbsynType() {
            return Absyn::kCompilationUnit;
        }
};

} //namespcae tamm

#endif



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
          std::string name;
          std::vector<std::string> upperIndices;
          std::vector<std::string> lowerIndices;
          std::string irrep;

          ArrayDeclaration(std::string n, std::vector<std::string>& ul, std::vector<std::string>& li){
              name = n;    
              upperIndices = ul;
              lowerIndices = li;
          }

          int getDeclType() {
              return Declaration::kArrayDeclaration;
          }
};

class IndexDeclaration : public Declaration {
    public:
          std::string name;
          std::string rangeID;

          IndexDeclaration(std::string n, std::string r){
              name = n;
              rangeID = r;
          }

          int getDeclType() {
              return Declaration::kIndexDeclaration;
          }
};

class RangeDeclaration : public Declaration {
    public:
          int value;
          std::string name;

          RangeDeclaration(std::string n, int v){
              name = n;    
              value = v;
          }

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
        std::string name;
        float coef;
        std::vector<std::string> indices;

        Array(std::string n, std::vector<std::string>& ind) {
            coef = 1.0;
            name = n;
            indices = ind;
        }

     int getExpressionType() { return Expression::kArrayRef; }
};

class AssignStatement: public Statement {
    public:
        Array* lhs;
        Expression* rhs;
        std::string label;
        std::string assign_op;

        AssignStatement(std::string assign_op, Array *lhs, Expression *rhs): assign_op(assign_op), lhs(lhs), rhs(rhs) {}
        
        int getStatementType() {
            return Statement::kAssignStatement;
        }

};



class DeclarationList: public Element {
public:
    std::vector<Declaration*> dlist;
    DeclarationList() {}
    DeclarationList(std::vector<Declaration*> &d): dlist(d) {}

    int getElementType() {
       return Element::kDeclarationList;
    }
};


class Parenth: public Expression {
    public:
        Expression *expression;
        float coef;

        Parenth(Expression* e) {
            coef = 1.0;
            expression = e;
        }

     int getExpressionType() { return Expression::kParenth; }
};

class NumConst: public Expression {
    public:
        float value;
        float coef;

        NumConst(float e) {
            value = e;
            coef = 1.0;
        }

     int getExpressionType() { return Expression::kNumConst; }
};



class Addition: public Expression {
    public:
        float coef;
        bool first_op;
        std::vector<Expression*> subexps;
        std::vector<std::string> add_operators;

    Addition(std::vector<Expression*>& se): subexps(se), first_op(false), coef(1.0) {}
    Addition(std::vector<Expression*>& se, std::vector<std::string> &ao, bool fop): subexps(se), add_operators(ao), first_op(fop), coef(1.0) {}
    int getExpressionType() { return Expression::kAddition; }

};



class Multiplication: public Expression {
    public:
        int coef;
        std::vector<Expression*> subexps;

        Multiplication(std::vector<Expression*>& se): subexps(se), coef(1.0) {}
        int getExpressionType() { return Expression::kMultiplication; }
    
};


class ElementList //group of declarations and statements corresponding to a single input
{
public:
    std::vector<Element*> elist;
    ElementList() {}
    ElementList(std::vector<Element*> &e): elist(e) {}

};


class Identifier: public Absyn {
public:

    std::string name;
    Identifier(std::string n){
        name = n;
    }

    int getAbsynType() {
       return Absyn::kIdentifier;
    }
};

class IdentifierList {
public:
    std::vector<Identifier*> idlist;
    IdentifierList() {}
    IdentifierList(std::vector<Identifier*> &d): idlist(d) {}
};


class CompoundElement  //represents a single input enclosed in { .. }
{
public:
    ElementList *elist;
    CompoundElement(ElementList *el): elist(el) {}

};

class CompilationUnit : public Absyn {
public:
    std::vector<CompoundElement*> celist;
    CompilationUnit(std::vector<CompoundElement*>& cel): celist(cel) {}

        int getAbsynType() {
            return Absyn::kCompilationUnit;
        }
};

} //namespcae tamm

#endif



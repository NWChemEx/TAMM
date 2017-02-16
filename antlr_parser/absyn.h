//
// Absyn
//  |
//  +-- TranslationUnit
//  |
//  +-- CompoundElement
//  |
//  +-- Element
//  |    |
//  |    +-- DeclarationList
//  |    |    |
//  |    |    +-- RangeDecl
//  |    |    +-- IndexDecl
//  |    |    +-- ArrayDecl
//  |    |    +-- ExpandDecl
//  |    |    +-- VolatileDecl
//  |    |    +-- IterationDecl
//  |    |
//  |    +-- StatementList
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

#ifndef ABSYN_H_
#define ABSYN_H_

#include "util.h"
#include <type_traits>

/* Forward Declarations */
class Declaration;
class Expression;
class DeclarationList;


/* The Absyn Hierarchy */

class Absyn //Root of the AST
{
    enum kAbsyn {
        kTranslationUnit, kCompoundElement, kElement, kIdentifier, kExpression
    };

};


class Declaration { //: public Absyn {
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

        /** Implemented in all direct subclasses.
         * Used to identify a direct subclass of Declaration.
         * Returns the enum value for the subclass
         * that calls this method.
         */
        virtual int getDeclType() = 0;

        // int getAbsynType() {
        //     return Absyn::kDeclaration;
        // }

        ~Declaration() { }
};


class ArrayDeclaration : public Declaration {
    public:
          int lineno;
          int pos;
          std::string name;
          std::vector<std::string> upperIndices;
          std::vector<std::string> lowerIndices;
          std::string irrep;

          ArrayDeclaration(std::string n, std::vector<std::string>& ul, std::vector<std::string>& li){
              lineno = 0;
              pos = 0;
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
          int lineno;
          int pos;
          std::string name;
          std::string rangeID;

          IndexDeclaration(std::string n, std::string r){
              lineno = 0;
              pos = 0;
              name = n;
              rangeID = r;
          }

          int getDeclType() {
              return Declaration::kIndexDeclaration;
          }
};

class RangeDeclaration : public Declaration {
    public:
          int lineno;
          int pos;

          int value;
          std::string name;
          

          RangeDeclaration(std::string n, int v){
              lineno = 0;
              pos = 0;
              name = n;    
              value = v;
          }

          int getDeclType() {
              return Declaration::kRangeDeclaration;
          }
};

class Element { //: public Absyn {
    public:
        /// An enum to identify direct subclasses of Element.
        enum kElement {
            kDeclarationList,
            kStatement
        };

        virtual int getElementType() = 0;

        // int getAbsynType() {
        //     return Absyn::kDeclaration;
        // }

        ~Element() { }
};


class Statement : public Element {
    public:
        /// An enum to identify direct subclasses of Element.
        enum kStatement {
            kAssignStatement
        };

        /** Implemented in all direct subclasses.
         * Used to identify a direct subclass of Element.
         * Returns the enum value for the subclass
         * that calls this method.
         */
        virtual int getStatementType() = 0;

        int getElementType() {
            return kElement::kStatement;
        }

        // int getAbsynType() {
        //     return Absyn::kDeclaration;
        // }

        ~Statement() { }
};

class AssignStatement: public Statement {
    public:
        Expression* lhs;
        Expression* rhs;
        int pos;
        std::string label;
        std::string astype;

        AssignStatement(Expression *lhs, Expression *rhs): lhs(lhs), rhs(rhs) {}
        
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

// class StatementList: public Element {
// public:
//     std::vector<Statement*> slist;
//     StatementList() {}
//     StatementList(std::vector<Statement*> &s): slist(s) {}

//     int getElementType() {
//        return Element::kStatementList;
//     }
// };

class Expression {
public:
    enum kExpression {
        kParenth, kNumConst, kArrayRef, kAddition, kMultiplication
    };

    virtual int getExpressionType() = 0;

    ~Expression() {}
};

class Parenth: public Expression {
    public:
        Expression *expression;
        int pos;
        int coef;
        int lineno;

        Parenth(Expression* e) {
            pos = 0;
            coef = 1;
            expression = e;
        }

     int getExpressionType() { return Expression::kParenth; }
};

class NumConst: public Expression {
    public:
        float value;
        int pos;
        int coef;
        int lineno;

        NumConst(float e) {
            pos = 0;
            coef = 1;
            value = e;
        }

     int getExpressionType() { return Expression::kNumConst; }
};


class Array: public Expression {
    public:
        std::string name;
        int pos;
        int coef;
        int lineno;
        std::vector<std::string> indices;

        Array(std::string n, std::vector<std::string>& ind) {
            pos = 0;
            coef = 1;
            name = n;
            indices = ind;
        }

     int getExpressionType() { return Expression::kArrayRef; }
};


class Addition: public Expression {
    public:
        int pos;
        int coef;
        int lineno;
        std::vector<Expression*> subexps;

    Addition(std::vector<Expression*>& se): subexps(se) {}
    int getExpressionType() { return Expression::kAddition; }

};



class Multiplication: public Expression {
    public:
        int pos;
        int coef;
        int lineno;
        std::vector<Expression*> subexps;

    Multiplication(std::vector<Expression*>& se): subexps(se) {}
    int getExpressionType() { return Expression::kMultiplication; }
    
};


class ElementList //group of declarations and statements corresponding to a single input
{
public:
    std::vector<Element*> elist;
    ElementList() {}
    ElementList(std::vector<Element*> &e): elist(e) {}
};


class Identifier {
public:
    int pos;
    int lineno;
    std::string name;

    Identifier(std::string n){
        pos = 0;
        name = n;
    }
};

class IdentifierList {
public:
    std::vector<Identifier*> idlist;
    IdentifierList() {}
    IdentifierList(std::vector<Identifier*> &d): idlist(d) {}
};

// class ExpressionList {
// public:
//     std::vector<Expression*> explist;
//     ExpressionList() {}
//     ExpressionList(std::vector<Expression*> &el): explist(el) {}

// };



class CompoundElement  //represents a single input enclosed in { .. }
{
public:
    ElementList *elist;
    CompoundElement(ElementList *el): elist(el) {}
};

class TranslationUnit {
public:
    std::vector<CompoundElement*> celist;
    TranslationUnit(std::vector<CompoundElement*>& cel): celist(cel) {}
};

#endif



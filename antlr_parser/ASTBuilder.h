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

#pragma once


#include "antlr4-runtime.h"
#include "TAMMVisitor.h"

#include "Absyn.h"

namespace tamm {

/**
 * This class provides an implementation of TAMMVisitor that builds a TAMM AST.
 */
class  ASTBuilder : public TAMMVisitor {
public:

 ASTBuilder() {}
~ASTBuilder() {}


  virtual antlrcpp::Any visitTranslation_unit(TAMMParser::Translation_unitContext *ctx) override {
    std::cout << "Enter translation unit\n";
    std::vector<CompoundElement*> cel = visit(ctx->children.at(0)); //Cleanup
    SymbolTable* const symbol_table = new SymbolTable();
    CompilationUnit *cu = new CompilationUnit(cel,symbol_table);
    return cu;
  }

  virtual antlrcpp::Any visitCompound_element_list(TAMMParser::Compound_element_listContext *ctx) override {
    std::cout << "Enter Compund Elem list\n";
    std::vector<CompoundElement*> cel;
    // Visit each compound element and add to list
    for (auto &ce: ctx->children) cel.push_back(visit(ce)); 
    return cel; 
  }

  virtual antlrcpp::Any visitCompound_element(TAMMParser::Compound_elementContext *ctx) override {
    std::cout << "Enter Compund Element \n";
    ElementList *get_el = nullptr;
    for (auto &x: ctx->children){
      if (TAMMParser::Element_listContext* t = dynamic_cast<TAMMParser::Element_listContext*>(x))
        get_el = visit(t);
    }
    return new CompoundElement(get_el);
  }

  virtual antlrcpp::Any visitElement_list(TAMMParser::Element_listContext *ctx) override {
    std::cout << "Enter Element List\n";
   std::vector<Element*> el;
    for (auto &elem: ctx->children) {
      el.push_back(visit(elem)); //returns Elem*
    }
    return new ElementList(el);  
  }

  virtual antlrcpp::Any visitElement(TAMMParser::ElementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDeclaration(TAMMParser::DeclarationContext *ctx) override {
    //Each declaration - index,range,etc returns a DeclarationList that is wrapped into an Elem type
    Element *e = visitChildren(ctx); //type Elem*
    return e;
  }

  virtual antlrcpp::Any visitScalar_declaration(TAMMParser::Scalar_declarationContext *ctx) override {
    assert (ctx->children.size() >= 2);
    std::vector<Declaration*> sdecls;
    
    for (auto &x: ctx->children) {
      if (TAMMParser::IdentifierContext* id = 
          dynamic_cast<TAMMParser::IdentifierContext*>(x)) {
        std::vector<Identifier*> u;
        std::vector<Identifier*> l;
        Identifier* i = visit(id);
        sdecls.push_back(new ArrayDeclaration(
                                  ctx->getStart()->getLine(),
                                  ctx->getStart()->getCharPositionInLine()+1,
                                  i->name,u,l));
      }
    }
     Element *s = new DeclarationList(ctx->getStart()->getLine(), sdecls);
     return s;
  }

  virtual antlrcpp::Any visitId_list_opt(TAMMParser::Id_list_optContext *ctx) override {
    /// Applies only to an Array Declaration
    if (ctx->children.size() == 0) {
      /// An Array can have 0 upper/lower indices
        std::vector<Identifier*> idlist;
        return new IdentifierList(idlist);
    }
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitId_list(TAMMParser::Id_listContext *ctx) override {
    std::vector<Identifier*> idlist;
    for (auto &x: ctx->children){
       if (TAMMParser::IdentifierContext* id = dynamic_cast<TAMMParser::IdentifierContext*>(x))
          idlist.push_back(visit(x));
    }
    return new IdentifierList(idlist);
  }

  /// Not used in grammar.
  virtual antlrcpp::Any visitNum_list(TAMMParser::Num_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIdentifier(TAMMParser::IdentifierContext *ctx) override {
    Identifier* id = new Identifier(ctx->getStart()->getLine(), 
                         ctx->getStart()->getCharPositionInLine()+1,
                         ctx->children.at(0)->getText());
    return id;
  }

  virtual antlrcpp::Any visitNumerical_constant(TAMMParser::Numerical_constantContext *ctx) override {

    std::string s = ctx->children.at(0)->getText();
    std::string delimiter = "/";

    size_t pos = 0;
    std::string numerator;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        numerator = s.substr(0, pos);
        s.erase(0, pos + delimiter.length());
    }
    
    float value = std::stof(s); // Gets denominator in case of fraction
    if (numerator.size() > 0) value = std::stof(numerator)*1.0/value;
    Expression* nc = new NumConst(ctx->getStart()->getLine(), 
                                  ctx->getStart()->getCharPositionInLine()+1, value);
    return nc;
  }

  virtual antlrcpp::Any visitRange_declaration(TAMMParser::Range_declarationContext *ctx) override {
    std::cout << "Enter Range Decl\n";
    std::vector<Declaration*> rd_list;

    int range_value = -1;
    IdentifierList* rnames = nullptr; //List of range variable names

      for (auto &x: ctx->children){
      if (TAMMParser::Id_listContext* id = dynamic_cast<TAMMParser::Id_listContext*>(x))
        rnames = visit(id);

      else if (TAMMParser::Numerical_constantContext* id = 
                dynamic_cast<TAMMParser::Numerical_constantContext*>(x)) {
         NumConst *nc = visit(id);
         range_value = (int)nc->value; 
      }
    }

    assert (range_value >= 0);
    assert (rnames != nullptr);

    for (auto &range: rnames->idlist)    {
      rd_list.push_back(new RangeDeclaration(ctx->getStart()->getLine(),
                              ctx->getStart()->getCharPositionInLine()+1, 
                              range->name, range_value));
    }

    //std::cout << "Leaving... Range Decl\n";
     Element *e = new DeclarationList(ctx->getStart()->getLine(), rd_list);
     return e;
  }

  virtual antlrcpp::Any visitIndex_declaration(TAMMParser::Index_declarationContext *ctx) override {
   std::cout << "Enter Index Decl\n";
    std::vector<Declaration*> id_list; //Store list of Index Declarations
  
    Identifier* range_var = nullptr;
    IdentifierList* inames = nullptr; //List of index names

    for (auto &x: ctx->children){
      if (TAMMParser::Id_listContext* id = dynamic_cast<TAMMParser::Id_listContext*>(x))
        inames = visit(id);

      else if (TAMMParser::IdentifierContext* id = 
                dynamic_cast<TAMMParser::IdentifierContext*>(x))
        range_var = visit(id);
    }

    assert (range_var != nullptr);
    assert (inames != nullptr);

    for (auto &index: inames->idlist)    {
      id_list.push_back(new IndexDeclaration(
                                  ctx->getStart()->getLine(),
                                  ctx->getStart()->getCharPositionInLine()+1,
                                  index->name, range_var->name));
    }

     Element *e = new DeclarationList(ctx->getStart()->getLine(), id_list);
     return e;
  }

  virtual antlrcpp::Any visitArray_declaration(TAMMParser::Array_declarationContext *ctx) override {
    std::cout << "Enter Array Decl\n";
    
    Element *adl;

    for (auto &x: ctx->children){
      if (TAMMParser::Array_structure_listContext* asl = 
            dynamic_cast<TAMMParser::Array_structure_listContext*>(x))
        adl = visit(x);
    }

    return adl;
  }



  virtual antlrcpp::Any visitArray_structure(TAMMParser::Array_structureContext *ctx) override {
    std::cout << "Enter array structure\n";
    bool ul_flag = true;
    IdentifierList* upper = nullptr;
    IdentifierList* lower = nullptr;

    std::string array_name = ctx->children.at(0)->getText();

    for (auto &x: ctx->children){
      if(TAMMParser::Id_list_optContext* ul 
            = dynamic_cast<TAMMParser::Id_list_optContext*>(x)) {
        if (ul_flag) { 
          upper = visit(ul);
          ul_flag = false;
        }
        else lower = visit(ul);
      }
    }

    assert(upper!=nullptr || lower!=nullptr);
    std::vector<Identifier*> ui;
    std::vector<Identifier*> li;
    if (upper != nullptr) ui = upper->idlist;
    if (lower != nullptr) li = lower->idlist;

    Declaration *d = new ArrayDeclaration(
                                  ctx->getStart()->getLine(),
                                  ctx->getStart()->getCharPositionInLine()+1,
                                  array_name, ui, li);
    return d;

  }

  virtual antlrcpp::Any visitArray_structure_list(TAMMParser::Array_structure_listContext *ctx) override {
    std::vector<Declaration*> ad;
    for (auto &x: ctx->children){
      if (TAMMParser::Array_structureContext* asl = 
            dynamic_cast<TAMMParser::Array_structureContext*>(x))
       ad.push_back(visit(asl));
    }

    Element *asl = new DeclarationList(ctx->getStart()->getLine(), ad);
    return asl;
  }

  virtual antlrcpp::Any visitStatement(TAMMParser::StatementContext *ctx) override {
    return visit(ctx->children.at(0));
  }


/// assignment_statement : (identifier COLON)? array_reference assignment_operator expression SEMI ;
  virtual antlrcpp::Any visitAssignment_statement(TAMMParser::Assignment_statementContext *ctx) override {
    std::cout << "Enter Assign Statement\n";

    std::string op_label;
    std::string assign_op;
    Array* lhs = nullptr;
    Expression* rhs = nullptr;

    for (auto &x: ctx->children){
      
      if  (TAMMParser::IdentifierContext* ic = dynamic_cast<TAMMParser::IdentifierContext*>(x))
        op_label = static_cast<Identifier*>(visit(x))->name;

      else if (TAMMParser::Array_referenceContext* ec = dynamic_cast<TAMMParser::Array_referenceContext*>(x)) {
        Expression *e = visit(x);
        if (Array* a = dynamic_cast<Array*>(e))   lhs = a;
        else ;// report error - this cannot happen
      }

      else if (TAMMParser::Assignment_operatorContext* op = dynamic_cast<TAMMParser::Assignment_operatorContext*>(x)) 
        assign_op = static_cast<Identifier*>(visit(x))->name;

      else if (TAMMParser::ExpressionContext* ec = dynamic_cast<TAMMParser::ExpressionContext*>(x)) 
        rhs = visit(x);
    }

    assert (assign_op.size() > 0);
    assert (lhs != nullptr && rhs != nullptr);

    const int line = ctx->getStart()->getLine();
    const int position = ctx->getStart()->getCharPositionInLine()+1;

    Element *e = nullptr; //Statement is child class of Element
    if (op_label.size() > 0) 
        e = new AssignStatement(line, position, op_label, assign_op,lhs,rhs);
    else e = new AssignStatement(line, position, assign_op,lhs,rhs); 
    return e;
  }

  virtual antlrcpp::Any visitAssignment_operator(TAMMParser::Assignment_operatorContext *ctx) override {
    Identifier *aop = new Identifier(ctx->getStart()->getLine(), 
                      ctx->getStart()->getCharPositionInLine()+1, 
                      ctx->children.at(0)->getText());
    return aop;
  }

  virtual antlrcpp::Any visitUnary_expression(TAMMParser::Unary_expressionContext *ctx) override {
    /// unary_expression :   numerical_constant | array_reference | ( expression )
    if (ctx->children.size() == 1) { return visit(ctx->children.at(0)); }
    else if (ctx->children.size() == 2) { return visit(ctx->children.at(1)); }
    else { 
      ; /// Error. This cannot happen since (expr) has the max 3 children (,expr,) 
    }
  }

  virtual antlrcpp::Any visitArray_reference(TAMMParser::Array_referenceContext *ctx) override {
    /// array_reference : ID (LBRACKET id_list RBRACKET)? 
    std::string name = ctx->children.at(0)->getText();
    IdentifierList *il = nullptr;
    
    for (auto &x: ctx->children){
      if(TAMMParser::Id_listContext* ul = dynamic_cast<TAMMParser::Id_listContext*>(x)){
          il = visit(ul);
      }
    }
    
    std::vector<Identifier*> indices;
    if (il!=nullptr) indices = il->idlist;
    Expression* ar = new Array(ctx->getStart()->getLine(),
                             ctx->getStart()->getCharPositionInLine()+1,
                             name,indices);
    return ar;
  }

  virtual antlrcpp::Any visitPlusORminus(TAMMParser::PlusORminusContext *ctx) override {
    return ctx->children.at(0)->getText();
  }


  virtual antlrcpp::Any visitExpression(TAMMParser::ExpressionContext *ctx) override {
    //Grammar: expression : (plusORminus)? multiplicative_expression (plusORminus multiplicative_expression)*

    // We only allow: c += alpha*a[]*b[] and c+= alpha * a[] for now

    //Default is an AddOP
    Expression *e = nullptr;
    std::vector<Expression*> am_ops;
    std::vector<std::string> signs;
    bool first_op_flag = false; //Check if the expression starts with a plus or minus sign

    if (TAMMParser::PlusORminusContext* pm = 
        dynamic_cast<TAMMParser::PlusORminusContext*>(ctx->children.at(0)))
      first_op_flag = true; 

    for (auto &x: ctx->children){
      //Has both add and mult ops, which in turn consist of NumConst and ArrayRefs
      if(TAMMParser::Multiplicative_expressionContext* me = 
              dynamic_cast<TAMMParser::Multiplicative_expressionContext*>(x))
        am_ops.push_back(visit(me)); 
      
      //The unary exps that have num consts get their signs from here.
      else if (TAMMParser::PlusORminusContext* pm = 
              dynamic_cast<TAMMParser::PlusORminusContext*>(x))
        signs.push_back(visit(x)); 
    }

    const int line = ctx->getStart()->getLine();
    const int position = ctx->getStart()->getCharPositionInLine()+1;
    
    e = new Addition(line, position, am_ops, signs, first_op_flag);
    return e;

  }

  virtual antlrcpp::Any visitMultiplicative_expression(TAMMParser::Multiplicative_expressionContext *ctx) override {
    /// Grammar: multiplicative_expression : unary_expression (TIMES unary_expression)*
    /// unary_expression :   numerical_constant | array_reference | ( expression )
    std::vector<Expression*> uexps;

    /// Get the Expression objects (NumConst, Array or Expression) returned by unary_expression
    for (auto &x: ctx->children){
      if(TAMMParser::Unary_expressionContext* me = dynamic_cast<TAMMParser::Unary_expressionContext*>(x))
        uexps.push_back(visit(me));
    }

/// We only allow: c += alpha*a[]*b[] and c+= alpha * a[] for now
    int num_array_refs = 0; ///< internally, scalar is also treated as tensor with 0 dims
    int num_consts = 0;

    std::vector<Expression*> trefs;

    /// Process the Expressions returned by unary_expression
    for (auto &t: uexps){
      if(NumConst* me = dynamic_cast<NumConst*>(t)){
        num_consts+=1; trefs.push_back(t);
      }
      else if (Array* me = dynamic_cast<Array*>(t))
       { num_array_refs +=1; trefs.push_back(t); }
      
      /// @todo Handle unary_expression = (expression) rule
      /// else if (TAMMParser::ExpressionContext* me = dynamic_cast<TAMMParser::ExpressionContext*>(x))
    }

    Expression* e = nullptr; 

    const int line = ctx->getStart()->getLine();
    const int position = ctx->getStart()->getCharPositionInLine()+1;

    assert (uexps.size() > 0 && uexps.size() <= 3);
    if (num_array_refs == 3 ) { ; /** Error cannot use scalar as a constant multiplier or cannot handle ternary operations; */     }
    if (num_consts == 2) { ; /** Error cannot use scalar as a constant multiplier; */     }

    /// Consts are also part of the Adds & Mults. Stored as NumConsts. 
    /// The sign for the consts is processed when processing the "Expression" rule later in intermediate code generation.
    if (num_array_refs == 1) {
      e = new Addition(line, position, trefs);  
    }
    else if (num_array_refs == 2){
      e = new Multiplication(line, position, trefs);
    }
    return e;
  }


};

}

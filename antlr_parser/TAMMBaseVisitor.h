
// Generated from TAMM.g4 by ANTLR 4.6

#pragma once


#include "antlr4-runtime.h"
#include "TAMMVisitor.h"

#include "absyn.h"

/**
 * This class provides an empty implementation of TAMMVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  TAMMBaseVisitor : public TAMMVisitor {
public:

 TAMMBaseVisitor() {}
~TAMMBaseVisitor() {}

  std::vector<std::string> getIdentifierList(std::vector<Identifier*> &idlist){
    std::vector<std::string> stringidlist;
    for (auto &id: idlist)
      stringidlist.push_back(id->name);
    return stringidlist;
  }


  virtual antlrcpp::Any visitTranslation_unit(TAMMParser::Translation_unitContext *ctx) override {
    std::cout << "Enter translation unit\n";
    std::vector<CompoundElement*> cel = visit(ctx->children.at(0)); //Cleanup
    TranslationUnit *tu = new TranslationUnit(cel);
    return tu;
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
      if (TAMMParser::IdentifierContext* id = dynamic_cast<TAMMParser::IdentifierContext*>(x)){
        std::vector<std::string> u;
        std::vector<std::string> l;
        Identifier* i = visit(id);
        sdecls.push_back(new ArrayDeclaration(i->name,u,l));
      }
    }
     Element *s = new DeclarationList(sdecls);
     return s;
  }

  virtual antlrcpp::Any visitId_list_opt(TAMMParser::Id_list_optContext *ctx) override {
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

  virtual antlrcpp::Any visitNum_list(TAMMParser::Num_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIdentifier(TAMMParser::IdentifierContext *ctx) override {
    Identifier* id = new Identifier(ctx->children.at(0)->getText());
    //id->lineno = tce_lineno;
    return id;
  }

  virtual antlrcpp::Any visitNumerical_constant(TAMMParser::Numerical_constantContext *ctx) override {
    return std::stoi(ctx->children.at(0)->getText());
  }

  virtual antlrcpp::Any visitRange_declaration(TAMMParser::Range_declarationContext *ctx) override {
    std::cout << "Enter Range Decl\n";
    std::vector<Declaration*> rd_list;

    int range_value = -1;
    IdentifierList* rnames = nullptr; //List of range variable names

      for (auto &x: ctx->children){
      if (TAMMParser::Id_listContext* id = dynamic_cast<TAMMParser::Id_listContext*>(x))
        rnames = visit(id);

      else if (TAMMParser::Numerical_constantContext* id = dynamic_cast<TAMMParser::Numerical_constantContext*>(x))
        range_value = visit(id);
    }

    assert (range_value >= 0);
    assert (rnames != nullptr);

    for (auto &range: rnames->idlist)    {
      rd_list.push_back(new RangeDeclaration(range->name, range_value));
    }

    //std::cout << "Leaving... Range Decl\n";
     Element *e = new DeclarationList(rd_list);
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

      else if (TAMMParser::IdentifierContext* id = dynamic_cast<TAMMParser::IdentifierContext*>(x))
        range_var = visit(id);
    }

    assert (range_var != nullptr);
    assert (inames != nullptr);

    for (auto &index: inames->idlist)    {
      id_list.push_back(new IndexDeclaration(index->name, range_var->name));
    }

     Element *e = new DeclarationList(id_list);
     return e;
  }

  virtual antlrcpp::Any visitArray_declaration(TAMMParser::Array_declarationContext *ctx) override {
    std::cout << "Enter Array Decl\n";
    
    Element *adl;

    for (auto &x: ctx->children){
      if (TAMMParser::Array_structure_listContext* asl = dynamic_cast<TAMMParser::Array_structure_listContext*>(x))
        adl = visit(x);
    }

    return adl;
  }



  virtual antlrcpp::Any visitArray_structure(TAMMParser::Array_structureContext *ctx) override {
    bool ul_flag = true;
    IdentifierList* upper = nullptr;
    IdentifierList* lower = nullptr;

    std::string array_name = ctx->children.at(0)->getText();

    for (auto &x: ctx->children){
      if(TAMMParser::Id_list_optContext* ul = dynamic_cast<TAMMParser::Id_list_optContext*>(x)){
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

    //C++14 - cannot pass getIdentifierList(..) directly as argument to array decl constructor - rvalue error
    std::vector<std::string> upper_indices = getIdentifierList(ui);
    std::vector<std::string> lower_indices = getIdentifierList(li);
    Declaration *d = new ArrayDeclaration(array_name, upper_indices, lower_indices);
    return d;

  }

  virtual antlrcpp::Any visitArray_structure_list(TAMMParser::Array_structure_listContext *ctx) override {
    std::vector<Declaration*> ad;
    for (auto &x: ctx->children){
      if (TAMMParser::Array_structureContext* asl = dynamic_cast<TAMMParser::Array_structureContext*>(x))
       ad.push_back(visit(asl));
    }

    Element *asl = new DeclarationList(ad);
    return asl;
  }

  virtual antlrcpp::Any visitStatement(TAMMParser::StatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAssignment_statement(TAMMParser::Assignment_statementContext *ctx) override {
    std::cout << "Enter Assign Statement\n";
    Element *e = new AssignStatement(nullptr,nullptr);
    return e;
  }

  virtual antlrcpp::Any visitAssignment_operator(TAMMParser::Assignment_operatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnary_expression(TAMMParser::Unary_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArray_reference(TAMMParser::Array_referenceContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPlusORminus(TAMMParser::PlusORminusContext *ctx) override {
    return ctx->children.at(0)->getText();
  }


  virtual antlrcpp::Any visitExpression(TAMMParser::ExpressionContext *ctx) override {
    //Grammar: expression : (plusORminus)? multiplicative_expression (plusORminus multiplicative_expression)*

    //Default is an AddOP
    Expression *e = nullptr;
    std::vector<Expression*> mults;
    std::vector<std::string> add_ops;

    bool first_op_flag = false; //Check if the expression starts with a plus or minus sign
    if (TAMMParser::PlusORminusContext* pm = dynamic_cast<TAMMParser::PlusORminusContext*>(ctx->children.at(0)))
      first_op_flag = true; 

    for (auto &x: ctx->children){
      if(TAMMParser::Multiplicative_expressionContext* me = dynamic_cast<TAMMParser::Multiplicative_expressionContext*>(x))
        mults.push_back(visit(me));
      
      else if (TAMMParser::PlusORminusContext* pm = dynamic_cast<TAMMParser::PlusORminusContext*>(x))
        add_ops.push_back(visit(x));
    }
    
    e = new Addition(mults, add_ops, first_op_flag);
    return e;

  }

  virtual antlrcpp::Any visitMultiplicative_expression(TAMMParser::Multiplicative_expressionContext *ctx) override {
    //Grammar: multiplicative_expression : unary_expression (TIMES unary_expression)*
    //         unary_expression :   numerical_constant | array_reference | ( expression )
    Expression *e = nullptr;
    std::vector<Expression*> uexps;

    for (auto &x: ctx->children){
      if(TAMMParser::Unary_expressionContext* me = dynamic_cast<TAMMParser::Unary_expressionContext*>(x))
        uexps.push_back(visit(me));
    }

    e = new Multiplication(uexps);
    return e;
  }


};


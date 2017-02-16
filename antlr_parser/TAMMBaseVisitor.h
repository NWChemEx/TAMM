
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
    std::cout << "Enter Element\n";
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDeclaration(TAMMParser::DeclarationContext *ctx) override {
    //Each declaration - index,range,etc returns a DeclarationList that is wrapped into an Elem type
    std::cout << "Enter Declaration\n";
    Element *e = visitChildren(ctx); //type Elem*
    return e;
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
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitRange_declaration(TAMMParser::Range_declarationContext *ctx) override {
    std::cout << "Enter Range Decl\n";
    auto rd_list = new DeclarationList();
    return rd_list;
  }

  virtual antlrcpp::Any visitIndex_declaration(TAMMParser::Index_declarationContext *ctx) override {
   std::cout << "Enter Index Decl\n";
    std::vector<Declaration*> id_list; //Store list of Index Declarations
  
    Identifier* range_var = nullptr;
    IdentifierList* inames = nullptr; //List of index names

    for (auto &x: ctx->children){
      if (TAMMParser::Id_listContext* id = dynamic_cast<TAMMParser::Id_listContext*>(x))
        inames = visit(id);

      else if (TAMMParser::IdentifierContext* ic = dynamic_cast<TAMMParser::IdentifierContext*>(x))
        range_var = visit(ic);

      //else visit(x);
    }

    assert (range_var != nullptr);
    assert (inames != nullptr);

    for (auto &index: inames->idlist)    {
      id_list.push_back(new IndexDeclaration(index->name, range_var->name));
    }

    std::cout << "Leaving... Index Decl\n";
     Element *e = new DeclarationList(id_list);
     return e;
  }

  virtual antlrcpp::Any visitArray_declaration(TAMMParser::Array_declarationContext *ctx) override {
    std::cout << "Enter Array Decl\n";
    Element *e = new DeclarationList();
    return e;
  }

  virtual antlrcpp::Any visitArray_structure(TAMMParser::Array_structureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArray_structure_list(TAMMParser::Array_structure_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPermut_symmetry(TAMMParser::Permut_symmetryContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSymmetry_group(TAMMParser::Symmetry_groupContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpansion_declaration(TAMMParser::Expansion_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVolatile_declaration(TAMMParser::Volatile_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIteration_declaration(TAMMParser::Iteration_declarationContext *ctx) override {
    return visitChildren(ctx);
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

  virtual antlrcpp::Any visitPrimary_expression(TAMMParser::Primary_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArray_reference(TAMMParser::Array_referenceContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPlusORminus(TAMMParser::PlusORminusContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpression(TAMMParser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMultiplicative_expression(TAMMParser::Multiplicative_expressionContext *ctx) override {
    return visitChildren(ctx);
  }


};



// Generated from /home/panyala/EclipseWS/workspacePTP/tamm/build/parser/TAMM.g4 by ANTLR 4.6

#pragma once


#include "antlr4-runtime.h"
#include "TAMMVisitor.h"

#include "absyn.h"
#include "intermediate.h"


/**
 * This class provides an empty implementation of TAMMVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  TAMMBaseVisitor : public TAMMVisitor {
public:

  Equations eqns;
  TAMMBaseVisitor(Equations &eqn){
    eqns = eqn;
  }
  ~TAMMBaseVisitor() { }

  virtual antlrcpp::Any visitTranslation_unit(TAMMParser::Translation_unitContext *ctx) override {
    std::cout << "Enter translation unit\n";
    //auto cel = visitChildren(ctx);
    //return new TranslationUnit(nullptr);
  }

  virtual antlrcpp::Any visitCompound_element(TAMMParser::Compound_elementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitElement(TAMMParser::ElementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDeclaration(TAMMParser::DeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitId_list_opt(TAMMParser::Id_list_optContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitId_list(TAMMParser::Id_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNum_list(TAMMParser::Num_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIdentifier(TAMMParser::IdentifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitNumerical_constant(TAMMParser::Numerical_constantContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitRange_declaration(TAMMParser::Range_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIndex_declaration(TAMMParser::Index_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArray_declaration(TAMMParser::Array_declarationContext *ctx) override {
    return visitChildren(ctx);
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
    return visitChildren(ctx);
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

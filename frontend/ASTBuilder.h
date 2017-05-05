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

#ifndef __TAMM_ASTBUILDER_H__
#define __TAMM_ASTBUILDER_H__

#include "TAMMVisitor.h"

#include "Absyn.h"
#include "Error.h"
#include "Util.h"

namespace tamm {

namespace frontend {

/**
 * This class provides an implementation of TAMMVisitor that builds a TAMM AST.
 */
class ASTBuilder : public TAMMVisitor {
 public:
  ASTBuilder() = default;
  ~ASTBuilder() override = default;

  antlrcpp::Any visitTranslation_unit(
      TAMMParser::Translation_unitContext *ctx) override {
    // std::cout << "Enter translation unit\n";
    std::vector<CompoundElement *> cel = visit(ctx->children.at(0));  // Cleanup
    auto *const symbol_table = new SymbolTable();
    CompilationUnit *cu = new CompilationUnit(cel, symbol_table);
    return cu;
  }

  antlrcpp::Any visitCompound_element_list(
      TAMMParser::Compound_element_listContext *ctx) override {
    // std::cout << "Enter Compund Element list\n";
    std::vector<CompoundElement *> cel;
    // Visit each compound element and add to list
    for (auto &ce : ctx->children) cel.push_back(visit(ce));
    return cel;
  }

  antlrcpp::Any visitCompound_element(
      TAMMParser::Compound_elementContext *ctx) override {
    // std::cout << "Enter Compund Element \n";
    ElementList *get_el = nullptr;
    for (auto &x : ctx->children) {
      if (TAMMParser::Element_listContext *t =
              dynamic_cast<TAMMParser::Element_listContext *>(x))
        get_el = visit(t);
    }
    return new CompoundElement(get_el);
  }

  antlrcpp::Any visitElement_list(
      TAMMParser::Element_listContext *ctx) override {
    // std::cout << "Enter Element List\n";
    std::vector<Element *> el;
    for (auto &elem : ctx->children) {
      el.push_back(visit(elem));  // returns Elem*
    }
    return new ElementList(el);
  }

  antlrcpp::Any visitElement(TAMMParser::ElementContext *ctx) override {
    return visitChildren(ctx);
  }

  antlrcpp::Any visitDeclaration(
      TAMMParser::DeclarationContext *ctx) override {
    // Each declaration - index,range,etc returns a DeclarationList that is
    // wrapped into an Elem type
    Element *e = visitChildren(ctx);  // type Elem*
    return e;
  }

  antlrcpp::Any visitScalar_declaration(
      TAMMParser::Scalar_declarationContext *ctx) override {
    assert(ctx->children.size() >= 2);
    std::vector<Declaration *> sdecls;

    for (auto &x : ctx->children) {
      if (TAMMParser::IdentifierContext *id =
              dynamic_cast<TAMMParser::IdentifierContext *>(x)) {
        std::vector<Identifier *> u;
        std::vector<Identifier *> l;
        Identifier *i = visit(id);
        sdecls.push_back(new ArrayDeclaration(
            ctx->getStart()->getLine(),
            ctx->getStart()->getCharPositionInLine() + 1, i, u, l, 1));
      }
    }
    Element *s = new DeclarationList(ctx->getStart()->getLine(), sdecls);
    return s;
  }

  antlrcpp::Any visitId_list_opt(
      TAMMParser::Id_list_optContext *ctx) override {
    /// Applies only to an Array Declaration
    if (ctx->children.size() == 0) {
      /// An Array can have 0 upper/lower indices
      std::vector<Identifier *> idlist;
      return new IdentifierList(idlist);
    }
    return visitChildren(ctx);
  }

  antlrcpp::Any visitId_list(TAMMParser::Id_listContext *ctx) override {
    std::vector<Identifier *> idlist;
    for (auto &x : ctx->children) {
      if (TAMMParser::IdentifierContext *id =
              dynamic_cast<TAMMParser::IdentifierContext *>(x))
        idlist.push_back(visit(x));
    }
    return new IdentifierList(idlist);
  }

  /// Not used in grammar.
  antlrcpp::Any visitNum_list(
      TAMMParser::Num_listContext *ctx) override {
    return visitChildren(ctx);
  }

  antlrcpp::Any visitIdentifier(
      TAMMParser::IdentifierContext *ctx) override {
    Identifier *id =
        new Identifier(ctx->getStart()->getLine(),
                       ctx->getStart()->getCharPositionInLine() + 1,
                       ctx->children.at(0)->getText());
    return id;
  }

  antlrcpp::Any visitInteger_constant(
      TAMMParser::Integer_constantContext *ctx) override {
    const int line = ctx->getStart()->getLine();
    const int position = ctx->getStart()->getCharPositionInLine() + 1;

    const std::string integer_const_error =
        "The range value must be a positive integer constant";
    if (ctx->children.size() == 0) Error(line, position, integer_const_error);

    const std::string s = ctx->children.at(0)->getText();
    if (!is_positive_integer(s)) Error(line, position, integer_const_error);

    int value = std::stoi(s);
    if (value <= 0) Error(line, position, integer_const_error);

    Expression *nc = new NumConst(line, position, value);
    return nc;
  }

  antlrcpp::Any visitNumerical_constant(
      TAMMParser::Numerical_constantContext *ctx) override {
    const int line = ctx->getStart()->getLine();
    const int position = ctx->getStart()->getCharPositionInLine() + 1;

    const std::string num_const_error =
        "A numerical constant must be either a positive, negative "
        "(integer/floating-point) constant or a fraction";
    if (ctx->children.size() == 0) Error(line, position, num_const_error);

    std::string s = ctx->children.at(0)->getText();
    std::string delimiter = "/";

    size_t pos = 0;
    std::string numerator;
    while ((pos = s.find(delimiter)) != std::string::npos) {
      numerator = s.substr(0, pos);
      s.erase(0, pos + delimiter.length());
    }

    float value = std::stof(s);  // Gets denominator in case of fraction
    if (numerator.size() > 0) value = std::stof(numerator) * 1.0 / value;

    /// @todo Can this happen - if (value <= 0) Error(line, position,
    /// num_const_error);
    Expression *nc = new NumConst(line, position, value);
    return nc;
  }

  antlrcpp::Any visitRange_declaration(
      TAMMParser::Range_declarationContext *ctx) override {
    // std::cout << "Enter Range Decl\n";
    std::vector<Declaration *> rd_list;
    const int line = ctx->getStart()->getLine();
    const int position = ctx->getStart()->getCharPositionInLine() + 1;

    int range_value = -1;
    IdentifierList *rnames = nullptr;  // List of range variable names

    for (auto &x : ctx->children) {
      if (TAMMParser::Id_listContext *id =
              dynamic_cast<TAMMParser::Id_listContext *>(x)) {
        const std::string range_variable = id->getText();
        // std::cout << range_variable << std::endl;
        if (range_variable.find("missing") != std::string::npos)
          Error(line, id->getStart()->getCharPositionInLine() + 1,
                "A range variable is not defined");
        rnames = visit(id);
      }

      else if (TAMMParser::Integer_constantContext *id =
                   dynamic_cast<TAMMParser::Integer_constantContext *>(x)) {
        Expression *const ncexp = visit(id);
        if (NumConst *const nc = dynamic_cast<NumConst *>(ncexp))
          range_value = (int)nc->value;
      }
    }

    /// assert (range_value >= 0);
    assert(rnames != nullptr);

    for (auto &range : rnames->idlist) {
      rd_list.push_back(
          new RangeDeclaration(line, position, range, range_value));
    }

    // std::cout << "Leaving... Range Decl\n";
    Element *e = new DeclarationList(line, rd_list);
    return e;
  }

  antlrcpp::Any visitIndex_declaration(
      TAMMParser::Index_declarationContext *ctx) override {
    // std::cout << "Enter Index Decl\n";
    std::vector<Declaration *> id_list;  // Store list of Index Declarations
    const int line = ctx->getStart()->getLine();
    const int position = ctx->getStart()->getCharPositionInLine() + 1;

    Identifier *range_var = nullptr;
    IdentifierList *inames = nullptr;  // List of index names

    for (auto &x : ctx->children) {
      if (TAMMParser::Id_listContext *id =
              dynamic_cast<TAMMParser::Id_listContext *>(x)) {
        inames = visit(id);
      }

      else if (TAMMParser::IdentifierContext *id =
                   dynamic_cast<TAMMParser::IdentifierContext *>(x))
        range_var = visit(id);
    }

    assert(range_var != nullptr);
    assert(inames != nullptr);

    for (auto &index : inames->idlist) {
      id_list.push_back(new IndexDeclaration(line, position, index, range_var));
    }

    Element *e = new DeclarationList(line, id_list);
    return e;
  }

  antlrcpp::Any visitArray_declaration(
      TAMMParser::Array_declarationContext *ctx) override {
    // std::cout << "Enter Array Decl\n";

    Element *adl = nullptr;

    for (auto &x : ctx->children) {
      if (TAMMParser::Array_structure_listContext *asl =
              dynamic_cast<TAMMParser::Array_structure_listContext *>(x)) 
        adl = visit(x);
    }
  return adl;
}

antlrcpp::Any
visitAuxbasis_id(TAMMParser::Auxbasis_idContext *ctx) override {
  //return visitChildren(ctx);
  return 1;
}

antlrcpp::Any visitArray_structure(
    TAMMParser::Array_structureContext *ctx) override {
  // std::cout << "Enter array structure\n";
  bool ul_flag = true;
  IdentifierList *upper = nullptr;
  IdentifierList *lower = nullptr;

  Identifier *array_name = visit(ctx->children.at(0));

  int auxbasis = 1;
  for (auto &x : ctx->children) {
    if (TAMMParser::Id_list_optContext *ul =
            dynamic_cast<TAMMParser::Id_list_optContext *>(x)) {
      if (ul_flag) {
        upper = visit(ul);
        ul_flag = false;
      } else
        lower = visit(ul);
    }
    if (TAMMParser::Auxbasis_idContext *aux =
            dynamic_cast<TAMMParser::Auxbasis_idContext *>(x)) {
              auxbasis = visit(x);
    }
  }

  assert(upper != nullptr || lower != nullptr);
  std::vector<Identifier *> ui;
  std::vector<Identifier *> li;
  if (upper != nullptr) ui = upper->idlist;
  if (lower != nullptr) li = lower->idlist;

  Declaration *d = new ArrayDeclaration(
      ctx->getStart()->getLine(), ctx->getStart()->getCharPositionInLine() + 1,
      array_name, ui, li, auxbasis);
  return d;
}

antlrcpp::Any visitArray_structure_list(
    TAMMParser::Array_structure_listContext *ctx) override {
  std::vector<Declaration *> ad;
  for (auto &x : ctx->children) {
    if (TAMMParser::Array_structureContext *asl =
            dynamic_cast<TAMMParser::Array_structureContext *>(x))
      ad.push_back(visit(asl));
  }

  Element *asl = new DeclarationList(ctx->getStart()->getLine(), ad);
  return asl;
}

antlrcpp::Any visitStatement(
    TAMMParser::StatementContext *ctx) override {
  return visit(ctx->children.at(0));
}

/// assignment_statement : (identifier COLON)? array_reference
/// assignment_operator expression SEMI ;
antlrcpp::Any visitAssignment_statement(
    TAMMParser::Assignment_statementContext *ctx) override {
  // std::cout << "Enter Assign Statement\n";

  std::string op_label;
  std::string assign_op;
  Array *lhs = nullptr;
  Expression *rhs = nullptr;

  const int line = ctx->getStart()->getLine();
  const int position = ctx->getStart()->getCharPositionInLine() + 1;

  for (auto &x : ctx->children) {
    if (TAMMParser::IdentifierContext *ic =
            dynamic_cast<TAMMParser::IdentifierContext *>(x))
      op_label = static_cast<Identifier *>(visit(x))->name;

    else if (TAMMParser::Array_referenceContext *ec =
                 dynamic_cast<TAMMParser::Array_referenceContext *>(x)) {
      Expression *e = visit(x);
      if (Array *a = dynamic_cast<Array *>(e)) lhs = a;
    }

    // else
    //   {
    //     const std::string lhs_exp_error = x->getText() + "\nLHS of an
    //     Assignment must be a Tensor reference\n";
    //     Error(line,position,lhs_exp_error);
    //   }

    else if (TAMMParser::Assignment_operatorContext *op =
                 dynamic_cast<TAMMParser::Assignment_operatorContext *>(x))
      assign_op = static_cast<Identifier *>(visit(x))->name;

    else if (TAMMParser::ExpressionContext *ec =
                 dynamic_cast<TAMMParser::ExpressionContext *>(x))
      rhs = visit(x);
  }

  assert(assign_op.size() > 0);
  assert(lhs != nullptr && rhs != nullptr);

  Element *e = nullptr;  // Statement is child class of Element
  if (op_label.size() > 0) {
    e = new AssignStatement(line, position, op_label, assign_op, lhs, rhs);
  } else {
    e = new AssignStatement(line, position, assign_op, lhs, rhs);
}
  return e;
}

antlrcpp::Any visitAssignment_operator(
    TAMMParser::Assignment_operatorContext *ctx) override {
  Identifier *const aop = new Identifier(
      ctx->getStart()->getLine(), ctx->getStart()->getCharPositionInLine() + 1,
      ctx->children.at(0)->getText());
  return aop;
}

antlrcpp::Any visitUnary_expression(
    TAMMParser::Unary_expressionContext *ctx) override {
  /// unary_expression :   numerical_constant | array_reference | ( expression
  /// )
  if (ctx->children.size() == 1) {
    return visit(ctx->children.at(0));
  } else if (ctx->children.size() == 2) {
    return visit(ctx->children.at(1));
  } else {
    /// @todo Is this ever triggered? This cannot happen since (expr) has the
    /// max 3 children (,expr,)
    Error(ctx->getStart()->getLine(),
          ctx->getStart()->getCharPositionInLine() + 1,
          "Malformed expression!");
  }
}

antlrcpp::Any visitArray_reference(
    TAMMParser::Array_referenceContext *ctx) override {
  /// array_reference : identifier (LBRACKET id_list RBRACKET)?

  Identifier *const name = visit(ctx->children.at(0));

  IdentifierList *il = nullptr;

  for (auto &x : ctx->children) {
    if (TAMMParser::Id_listContext *ul =
            dynamic_cast<TAMMParser::Id_listContext *>(x)) {
      il = visit(ul);
    }
  }

  std::vector<Identifier *> indices;
  if (il != nullptr) indices = il->idlist;
  Expression *ar = new Array(ctx->getStart()->getLine(),
                             ctx->getStart()->getCharPositionInLine() + 1,
                             ctx->getText(), name, indices);
  return ar;
}

antlrcpp::Any visitPlusORminus(
    TAMMParser::PlusORminusContext *ctx) override {
  return ctx->children.at(0)->getText();
}

antlrcpp::Any visitExpression(
    TAMMParser::ExpressionContext *ctx) override {
  // Grammar: expression : (plusORminus)? multiplicative_expression
  // (plusORminus multiplicative_expression)*

  // We only allow: c += alpha*a[]*b[] and c+= alpha * a[] for now

  // Default is an AddOP
  Expression *e = nullptr;
  std::vector<Expression *> am_ops;
  std::vector<std::string> signs;
  bool first_op_flag =
      false;  // Check if the expression starts with a plus or minus sign

  if (TAMMParser::PlusORminusContext *pm =
          dynamic_cast<TAMMParser::PlusORminusContext *>(ctx->children.at(0)))
    first_op_flag = true;

  for (auto &x : ctx->children) {
    // Has both add and mult ops, which in turn consist of NumConst and
    // ArrayRefs
    if (TAMMParser::Multiplicative_expressionContext *me =
            dynamic_cast<TAMMParser::Multiplicative_expressionContext *>(x))
      am_ops.push_back(visit(me));

    // The unary exps that have num consts get their signs from here.
    else if (TAMMParser::PlusORminusContext *pm =
                 dynamic_cast<TAMMParser::PlusORminusContext *>(x))
      signs.push_back(visit(x));
  }

  const int line = ctx->getStart()->getLine();
  const int position = ctx->getStart()->getCharPositionInLine() + 1;

  e = new Addition(line, position, am_ops, signs, first_op_flag);
  return e;
}

antlrcpp::Any visitMultiplicative_expression(
    TAMMParser::Multiplicative_expressionContext *ctx) override {
  /// Grammar: multiplicative_expression : unary_expression (TIMES
  /// unary_expression)* unary_expression :   numerical_constant |
  /// array_reference | ( expression )
  std::vector<Expression *> uexps;

  /// Get the Expression objects (NumConst, Array or Expression) returned by
  /// unary_expression
  for (auto &x : ctx->children) {
    if (TAMMParser::Unary_expressionContext *me =
            dynamic_cast<TAMMParser::Unary_expressionContext *>(x))
      uexps.push_back(visit(me));
  }

  /// We only allow: c += alpha*a[]*b[] and c+= alpha * a[] for now
  int num_array_refs =
      0;  ///< internally, scalar is also treated as tensor with 0 dims
  int num_consts = 0;

  std::vector<Expression *> trefs;

  /// Process the Expressions returned by unary_expression
  for (auto &t : uexps) {
    if (NumConst *me = dynamic_cast<NumConst *>(t)) {
      num_consts += 1;
      trefs.push_back(t);
    } else if (Array *me = dynamic_cast<Array *>(t)) {
      num_array_refs += 1;
      trefs.push_back(t);
    }

    /// @todo Handle unary_expression = (expression) rule
    /// else if (TAMMParser::ExpressionContext* me =
    /// dynamic_cast<TAMMParser::ExpressionContext*>(x))
  }

  Expression *e = nullptr;

  const int line = ctx->getStart()->getLine();
  const int position = ctx->getStart()->getCharPositionInLine() + 1;

  assert(uexps.size() > 0 && uexps.size() <= 3);
  if (num_array_refs == 3) {
    ; /** Error cannot use scalar as a constant multiplier or cannot handle
         ternary operations; */
  }
  if (num_consts == 2) {
    ; /** Error cannot use scalar as a constant multiplier; */
  }

  /// Consts are also part of the Adds & Mults. Stored as NumConsts.
  /// The sign for the consts is processed when processing the "Expression"
  /// rule later in intermediate code generation.
  if (num_array_refs == 1) {
    e = new Addition(line, position, trefs);
  } else if (num_array_refs == 2) {
    e = new Multiplication(line, position, trefs);
  }
  return e;
}
};  // class ASTBuilder

}  // namespace frontend
}  // namespace tamm

#endif
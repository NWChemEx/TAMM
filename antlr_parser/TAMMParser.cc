
// Generated from TAMM.g4 by ANTLR 4.6


#include "TAMMVisitor.h"

#include "TAMMParser.h"


using namespace antlrcpp;
using namespace antlr4;

TAMMParser::TAMMParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

TAMMParser::~TAMMParser() {
  delete _interpreter;
}

std::string TAMMParser::getGrammarFileName() const {
  return "TAMM.g4";
}

const std::vector<std::string>& TAMMParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& TAMMParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- Translation_unitContext ------------------------------------------------------------------

TAMMParser::Translation_unitContext::Translation_unitContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAMMParser::Compound_element_listContext* TAMMParser::Translation_unitContext::compound_element_list() {
  return getRuleContext<TAMMParser::Compound_element_listContext>(0);
}

tree::TerminalNode* TAMMParser::Translation_unitContext::EOF() {
  return getToken(TAMMParser::EOF, 0);
}


size_t TAMMParser::Translation_unitContext::getRuleIndex() const {
  return TAMMParser::RuleTranslation_unit;
}

antlrcpp::Any TAMMParser::Translation_unitContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitTranslation_unit(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Translation_unitContext* TAMMParser::translation_unit() {
  Translation_unitContext *_localctx = _tracker.createInstance<Translation_unitContext>(_ctx, getState());
  enterRule(_localctx, 0, TAMMParser::RuleTranslation_unit);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(60);
    compound_element_list();
    setState(61);
    match(TAMMParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Compound_element_listContext ------------------------------------------------------------------

TAMMParser::Compound_element_listContext::Compound_element_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAMMParser::Compound_elementContext *> TAMMParser::Compound_element_listContext::compound_element() {
  return getRuleContexts<TAMMParser::Compound_elementContext>();
}

TAMMParser::Compound_elementContext* TAMMParser::Compound_element_listContext::compound_element(size_t i) {
  return getRuleContext<TAMMParser::Compound_elementContext>(i);
}


size_t TAMMParser::Compound_element_listContext::getRuleIndex() const {
  return TAMMParser::RuleCompound_element_list;
}

antlrcpp::Any TAMMParser::Compound_element_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitCompound_element_list(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Compound_element_listContext* TAMMParser::compound_element_list() {
  Compound_element_listContext *_localctx = _tracker.createInstance<Compound_element_listContext>(_ctx, getState());
  enterRule(_localctx, 2, TAMMParser::RuleCompound_element_list);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(66);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TAMMParser::ID) {
      setState(63);
      compound_element();
      setState(68);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Element_listContext ------------------------------------------------------------------

TAMMParser::Element_listContext::Element_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAMMParser::ElementContext *> TAMMParser::Element_listContext::element() {
  return getRuleContexts<TAMMParser::ElementContext>();
}

TAMMParser::ElementContext* TAMMParser::Element_listContext::element(size_t i) {
  return getRuleContext<TAMMParser::ElementContext>(i);
}


size_t TAMMParser::Element_listContext::getRuleIndex() const {
  return TAMMParser::RuleElement_list;
}

antlrcpp::Any TAMMParser::Element_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitElement_list(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Element_listContext* TAMMParser::element_list() {
  Element_listContext *_localctx = _tracker.createInstance<Element_listContext>(_ctx, getState());
  enterRule(_localctx, 4, TAMMParser::RuleElement_list);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(72);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << TAMMParser::RANGE)
      | (1ULL << TAMMParser::INDEX)
      | (1ULL << TAMMParser::ARRAY)
      | (1ULL << TAMMParser::EXPAND)
      | (1ULL << TAMMParser::VOLATILE)
      | (1ULL << TAMMParser::ITERATION)
      | (1ULL << TAMMParser::PLUS)
      | (1ULL << TAMMParser::MINUS)
      | (1ULL << TAMMParser::LPAREN)
      | (1ULL << TAMMParser::ID)
      | (1ULL << TAMMParser::ICONST)
      | (1ULL << TAMMParser::FRAC)
      | (1ULL << TAMMParser::FCONST))) != 0)) {
      setState(69);
      element();
      setState(74);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Compound_elementContext ------------------------------------------------------------------

TAMMParser::Compound_elementContext::Compound_elementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAMMParser::IdentifierContext* TAMMParser::Compound_elementContext::identifier() {
  return getRuleContext<TAMMParser::IdentifierContext>(0);
}

tree::TerminalNode* TAMMParser::Compound_elementContext::LBRACE() {
  return getToken(TAMMParser::LBRACE, 0);
}

TAMMParser::Element_listContext* TAMMParser::Compound_elementContext::element_list() {
  return getRuleContext<TAMMParser::Element_listContext>(0);
}

tree::TerminalNode* TAMMParser::Compound_elementContext::RBRACE() {
  return getToken(TAMMParser::RBRACE, 0);
}


size_t TAMMParser::Compound_elementContext::getRuleIndex() const {
  return TAMMParser::RuleCompound_element;
}

antlrcpp::Any TAMMParser::Compound_elementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitCompound_element(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Compound_elementContext* TAMMParser::compound_element() {
  Compound_elementContext *_localctx = _tracker.createInstance<Compound_elementContext>(_ctx, getState());
  enterRule(_localctx, 6, TAMMParser::RuleCompound_element);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(75);
    identifier();
    setState(76);
    match(TAMMParser::LBRACE);
    setState(77);
    element_list();
    setState(78);
    match(TAMMParser::RBRACE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ElementContext ------------------------------------------------------------------

TAMMParser::ElementContext::ElementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAMMParser::DeclarationContext* TAMMParser::ElementContext::declaration() {
  return getRuleContext<TAMMParser::DeclarationContext>(0);
}

TAMMParser::StatementContext* TAMMParser::ElementContext::statement() {
  return getRuleContext<TAMMParser::StatementContext>(0);
}


size_t TAMMParser::ElementContext::getRuleIndex() const {
  return TAMMParser::RuleElement;
}

antlrcpp::Any TAMMParser::ElementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitElement(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::ElementContext* TAMMParser::element() {
  ElementContext *_localctx = _tracker.createInstance<ElementContext>(_ctx, getState());
  enterRule(_localctx, 8, TAMMParser::RuleElement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(82);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAMMParser::RANGE:
      case TAMMParser::INDEX:
      case TAMMParser::ARRAY:
      case TAMMParser::EXPAND:
      case TAMMParser::VOLATILE:
      case TAMMParser::ITERATION: {
        enterOuterAlt(_localctx, 1);
        setState(80);
        declaration();
        break;
      }

      case TAMMParser::PLUS:
      case TAMMParser::MINUS:
      case TAMMParser::LPAREN:
      case TAMMParser::ID:
      case TAMMParser::ICONST:
      case TAMMParser::FRAC:
      case TAMMParser::FCONST: {
        enterOuterAlt(_localctx, 2);
        setState(81);
        statement();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DeclarationContext ------------------------------------------------------------------

TAMMParser::DeclarationContext::DeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAMMParser::Range_declarationContext* TAMMParser::DeclarationContext::range_declaration() {
  return getRuleContext<TAMMParser::Range_declarationContext>(0);
}

TAMMParser::Index_declarationContext* TAMMParser::DeclarationContext::index_declaration() {
  return getRuleContext<TAMMParser::Index_declarationContext>(0);
}

TAMMParser::Array_declarationContext* TAMMParser::DeclarationContext::array_declaration() {
  return getRuleContext<TAMMParser::Array_declarationContext>(0);
}

TAMMParser::Expansion_declarationContext* TAMMParser::DeclarationContext::expansion_declaration() {
  return getRuleContext<TAMMParser::Expansion_declarationContext>(0);
}

TAMMParser::Volatile_declarationContext* TAMMParser::DeclarationContext::volatile_declaration() {
  return getRuleContext<TAMMParser::Volatile_declarationContext>(0);
}

TAMMParser::Iteration_declarationContext* TAMMParser::DeclarationContext::iteration_declaration() {
  return getRuleContext<TAMMParser::Iteration_declarationContext>(0);
}


size_t TAMMParser::DeclarationContext::getRuleIndex() const {
  return TAMMParser::RuleDeclaration;
}

antlrcpp::Any TAMMParser::DeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitDeclaration(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::DeclarationContext* TAMMParser::declaration() {
  DeclarationContext *_localctx = _tracker.createInstance<DeclarationContext>(_ctx, getState());
  enterRule(_localctx, 10, TAMMParser::RuleDeclaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(90);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAMMParser::RANGE: {
        enterOuterAlt(_localctx, 1);
        setState(84);
        range_declaration();
        break;
      }

      case TAMMParser::INDEX: {
        enterOuterAlt(_localctx, 2);
        setState(85);
        index_declaration();
        break;
      }

      case TAMMParser::ARRAY: {
        enterOuterAlt(_localctx, 3);
        setState(86);
        array_declaration();
        break;
      }

      case TAMMParser::EXPAND: {
        enterOuterAlt(_localctx, 4);
        setState(87);
        expansion_declaration();
        break;
      }

      case TAMMParser::VOLATILE: {
        enterOuterAlt(_localctx, 5);
        setState(88);
        volatile_declaration();
        break;
      }

      case TAMMParser::ITERATION: {
        enterOuterAlt(_localctx, 6);
        setState(89);
        iteration_declaration();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Id_list_optContext ------------------------------------------------------------------

TAMMParser::Id_list_optContext::Id_list_optContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAMMParser::Id_listContext* TAMMParser::Id_list_optContext::id_list() {
  return getRuleContext<TAMMParser::Id_listContext>(0);
}


size_t TAMMParser::Id_list_optContext::getRuleIndex() const {
  return TAMMParser::RuleId_list_opt;
}

antlrcpp::Any TAMMParser::Id_list_optContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitId_list_opt(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Id_list_optContext* TAMMParser::id_list_opt() {
  Id_list_optContext *_localctx = _tracker.createInstance<Id_list_optContext>(_ctx, getState());
  enterRule(_localctx, 12, TAMMParser::RuleId_list_opt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(94);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAMMParser::RBRACKET: {
        enterOuterAlt(_localctx, 1);

        break;
      }

      case TAMMParser::ID: {
        enterOuterAlt(_localctx, 2);
        setState(93);
        id_list();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Id_listContext ------------------------------------------------------------------

TAMMParser::Id_listContext::Id_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAMMParser::IdentifierContext *> TAMMParser::Id_listContext::identifier() {
  return getRuleContexts<TAMMParser::IdentifierContext>();
}

TAMMParser::IdentifierContext* TAMMParser::Id_listContext::identifier(size_t i) {
  return getRuleContext<TAMMParser::IdentifierContext>(i);
}

std::vector<tree::TerminalNode *> TAMMParser::Id_listContext::COMMA() {
  return getTokens(TAMMParser::COMMA);
}

tree::TerminalNode* TAMMParser::Id_listContext::COMMA(size_t i) {
  return getToken(TAMMParser::COMMA, i);
}


size_t TAMMParser::Id_listContext::getRuleIndex() const {
  return TAMMParser::RuleId_list;
}

antlrcpp::Any TAMMParser::Id_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitId_list(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Id_listContext* TAMMParser::id_list() {
  Id_listContext *_localctx = _tracker.createInstance<Id_listContext>(_ctx, getState());
  enterRule(_localctx, 14, TAMMParser::RuleId_list);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(96);
    identifier();
    setState(101);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TAMMParser::COMMA) {
      setState(97);
      match(TAMMParser::COMMA);
      setState(98);
      identifier();
      setState(103);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Num_listContext ------------------------------------------------------------------

TAMMParser::Num_listContext::Num_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAMMParser::Numerical_constantContext *> TAMMParser::Num_listContext::numerical_constant() {
  return getRuleContexts<TAMMParser::Numerical_constantContext>();
}

TAMMParser::Numerical_constantContext* TAMMParser::Num_listContext::numerical_constant(size_t i) {
  return getRuleContext<TAMMParser::Numerical_constantContext>(i);
}

std::vector<tree::TerminalNode *> TAMMParser::Num_listContext::COMMA() {
  return getTokens(TAMMParser::COMMA);
}

tree::TerminalNode* TAMMParser::Num_listContext::COMMA(size_t i) {
  return getToken(TAMMParser::COMMA, i);
}


size_t TAMMParser::Num_listContext::getRuleIndex() const {
  return TAMMParser::RuleNum_list;
}

antlrcpp::Any TAMMParser::Num_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitNum_list(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Num_listContext* TAMMParser::num_list() {
  Num_listContext *_localctx = _tracker.createInstance<Num_listContext>(_ctx, getState());
  enterRule(_localctx, 16, TAMMParser::RuleNum_list);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(104);
    numerical_constant();
    setState(109);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TAMMParser::COMMA) {
      setState(105);
      match(TAMMParser::COMMA);
      setState(106);
      numerical_constant();
      setState(111);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdentifierContext ------------------------------------------------------------------

TAMMParser::IdentifierContext::IdentifierContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::IdentifierContext::ID() {
  return getToken(TAMMParser::ID, 0);
}


size_t TAMMParser::IdentifierContext::getRuleIndex() const {
  return TAMMParser::RuleIdentifier;
}

antlrcpp::Any TAMMParser::IdentifierContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitIdentifier(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::IdentifierContext* TAMMParser::identifier() {
  IdentifierContext *_localctx = _tracker.createInstance<IdentifierContext>(_ctx, getState());
  enterRule(_localctx, 18, TAMMParser::RuleIdentifier);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(112);
    match(TAMMParser::ID);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Numerical_constantContext ------------------------------------------------------------------

TAMMParser::Numerical_constantContext::Numerical_constantContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Numerical_constantContext::ICONST() {
  return getToken(TAMMParser::ICONST, 0);
}

tree::TerminalNode* TAMMParser::Numerical_constantContext::FCONST() {
  return getToken(TAMMParser::FCONST, 0);
}

tree::TerminalNode* TAMMParser::Numerical_constantContext::FRAC() {
  return getToken(TAMMParser::FRAC, 0);
}


size_t TAMMParser::Numerical_constantContext::getRuleIndex() const {
  return TAMMParser::RuleNumerical_constant;
}

antlrcpp::Any TAMMParser::Numerical_constantContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitNumerical_constant(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Numerical_constantContext* TAMMParser::numerical_constant() {
  Numerical_constantContext *_localctx = _tracker.createInstance<Numerical_constantContext>(_ctx, getState());
  enterRule(_localctx, 20, TAMMParser::RuleNumerical_constant);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(114);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << TAMMParser::ICONST)
      | (1ULL << TAMMParser::FRAC)
      | (1ULL << TAMMParser::FCONST))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Range_declarationContext ------------------------------------------------------------------

TAMMParser::Range_declarationContext::Range_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Range_declarationContext::RANGE() {
  return getToken(TAMMParser::RANGE, 0);
}

TAMMParser::Id_listContext* TAMMParser::Range_declarationContext::id_list() {
  return getRuleContext<TAMMParser::Id_listContext>(0);
}

tree::TerminalNode* TAMMParser::Range_declarationContext::EQUALS() {
  return getToken(TAMMParser::EQUALS, 0);
}

TAMMParser::Numerical_constantContext* TAMMParser::Range_declarationContext::numerical_constant() {
  return getRuleContext<TAMMParser::Numerical_constantContext>(0);
}

tree::TerminalNode* TAMMParser::Range_declarationContext::SEMI() {
  return getToken(TAMMParser::SEMI, 0);
}


size_t TAMMParser::Range_declarationContext::getRuleIndex() const {
  return TAMMParser::RuleRange_declaration;
}

antlrcpp::Any TAMMParser::Range_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitRange_declaration(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Range_declarationContext* TAMMParser::range_declaration() {
  Range_declarationContext *_localctx = _tracker.createInstance<Range_declarationContext>(_ctx, getState());
  enterRule(_localctx, 22, TAMMParser::RuleRange_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(116);
    match(TAMMParser::RANGE);
    setState(117);
    id_list();
    setState(118);
    match(TAMMParser::EQUALS);
    setState(119);
    numerical_constant();
    setState(120);
    match(TAMMParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Index_declarationContext ------------------------------------------------------------------

TAMMParser::Index_declarationContext::Index_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Index_declarationContext::INDEX() {
  return getToken(TAMMParser::INDEX, 0);
}

TAMMParser::Id_listContext* TAMMParser::Index_declarationContext::id_list() {
  return getRuleContext<TAMMParser::Id_listContext>(0);
}

tree::TerminalNode* TAMMParser::Index_declarationContext::EQUALS() {
  return getToken(TAMMParser::EQUALS, 0);
}

TAMMParser::IdentifierContext* TAMMParser::Index_declarationContext::identifier() {
  return getRuleContext<TAMMParser::IdentifierContext>(0);
}

tree::TerminalNode* TAMMParser::Index_declarationContext::SEMI() {
  return getToken(TAMMParser::SEMI, 0);
}


size_t TAMMParser::Index_declarationContext::getRuleIndex() const {
  return TAMMParser::RuleIndex_declaration;
}

antlrcpp::Any TAMMParser::Index_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitIndex_declaration(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Index_declarationContext* TAMMParser::index_declaration() {
  Index_declarationContext *_localctx = _tracker.createInstance<Index_declarationContext>(_ctx, getState());
  enterRule(_localctx, 24, TAMMParser::RuleIndex_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(122);
    match(TAMMParser::INDEX);
    setState(123);
    id_list();
    setState(124);
    match(TAMMParser::EQUALS);
    setState(125);
    identifier();
    setState(126);
    match(TAMMParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Array_declarationContext ------------------------------------------------------------------

TAMMParser::Array_declarationContext::Array_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Array_declarationContext::ARRAY() {
  return getToken(TAMMParser::ARRAY, 0);
}

TAMMParser::Array_structure_listContext* TAMMParser::Array_declarationContext::array_structure_list() {
  return getRuleContext<TAMMParser::Array_structure_listContext>(0);
}

tree::TerminalNode* TAMMParser::Array_declarationContext::SEMI() {
  return getToken(TAMMParser::SEMI, 0);
}

tree::TerminalNode* TAMMParser::Array_declarationContext::COLON() {
  return getToken(TAMMParser::COLON, 0);
}

TAMMParser::IdentifierContext* TAMMParser::Array_declarationContext::identifier() {
  return getRuleContext<TAMMParser::IdentifierContext>(0);
}


size_t TAMMParser::Array_declarationContext::getRuleIndex() const {
  return TAMMParser::RuleArray_declaration;
}

antlrcpp::Any TAMMParser::Array_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitArray_declaration(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Array_declarationContext* TAMMParser::array_declaration() {
  Array_declarationContext *_localctx = _tracker.createInstance<Array_declarationContext>(_ctx, getState());
  enterRule(_localctx, 26, TAMMParser::RuleArray_declaration);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(128);
    match(TAMMParser::ARRAY);
    setState(129);
    array_structure_list();
    setState(132);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAMMParser::COLON) {
      setState(130);
      match(TAMMParser::COLON);
      setState(131);
      identifier();
    }
    setState(134);
    match(TAMMParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Array_structureContext ------------------------------------------------------------------

TAMMParser::Array_structureContext::Array_structureContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Array_structureContext::ID() {
  return getToken(TAMMParser::ID, 0);
}

std::vector<tree::TerminalNode *> TAMMParser::Array_structureContext::LBRACKET() {
  return getTokens(TAMMParser::LBRACKET);
}

tree::TerminalNode* TAMMParser::Array_structureContext::LBRACKET(size_t i) {
  return getToken(TAMMParser::LBRACKET, i);
}

std::vector<TAMMParser::Id_list_optContext *> TAMMParser::Array_structureContext::id_list_opt() {
  return getRuleContexts<TAMMParser::Id_list_optContext>();
}

TAMMParser::Id_list_optContext* TAMMParser::Array_structureContext::id_list_opt(size_t i) {
  return getRuleContext<TAMMParser::Id_list_optContext>(i);
}

std::vector<tree::TerminalNode *> TAMMParser::Array_structureContext::RBRACKET() {
  return getTokens(TAMMParser::RBRACKET);
}

tree::TerminalNode* TAMMParser::Array_structureContext::RBRACKET(size_t i) {
  return getToken(TAMMParser::RBRACKET, i);
}

TAMMParser::Permut_symmetryContext* TAMMParser::Array_structureContext::permut_symmetry() {
  return getRuleContext<TAMMParser::Permut_symmetryContext>(0);
}


size_t TAMMParser::Array_structureContext::getRuleIndex() const {
  return TAMMParser::RuleArray_structure;
}

antlrcpp::Any TAMMParser::Array_structureContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitArray_structure(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Array_structureContext* TAMMParser::array_structure() {
  Array_structureContext *_localctx = _tracker.createInstance<Array_structureContext>(_ctx, getState());
  enterRule(_localctx, 28, TAMMParser::RuleArray_structure);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(136);
    match(TAMMParser::ID);
    setState(137);
    match(TAMMParser::LBRACKET);
    setState(138);
    id_list_opt();
    setState(139);
    match(TAMMParser::RBRACKET);
    setState(140);
    match(TAMMParser::LBRACKET);
    setState(141);
    id_list_opt();
    setState(142);
    match(TAMMParser::RBRACKET);
    setState(144);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      setState(143);
      permut_symmetry();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Array_structure_listContext ------------------------------------------------------------------

TAMMParser::Array_structure_listContext::Array_structure_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAMMParser::Array_structureContext *> TAMMParser::Array_structure_listContext::array_structure() {
  return getRuleContexts<TAMMParser::Array_structureContext>();
}

TAMMParser::Array_structureContext* TAMMParser::Array_structure_listContext::array_structure(size_t i) {
  return getRuleContext<TAMMParser::Array_structureContext>(i);
}

std::vector<tree::TerminalNode *> TAMMParser::Array_structure_listContext::COMMA() {
  return getTokens(TAMMParser::COMMA);
}

tree::TerminalNode* TAMMParser::Array_structure_listContext::COMMA(size_t i) {
  return getToken(TAMMParser::COMMA, i);
}


size_t TAMMParser::Array_structure_listContext::getRuleIndex() const {
  return TAMMParser::RuleArray_structure_list;
}

antlrcpp::Any TAMMParser::Array_structure_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitArray_structure_list(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Array_structure_listContext* TAMMParser::array_structure_list() {
  Array_structure_listContext *_localctx = _tracker.createInstance<Array_structure_listContext>(_ctx, getState());
  enterRule(_localctx, 30, TAMMParser::RuleArray_structure_list);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(146);
    array_structure();
    setState(151);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TAMMParser::COMMA) {
      setState(147);
      match(TAMMParser::COMMA);
      setState(148);
      array_structure();
      setState(153);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Permut_symmetryContext ------------------------------------------------------------------

TAMMParser::Permut_symmetryContext::Permut_symmetryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Permut_symmetryContext::COLON() {
  return getToken(TAMMParser::COLON, 0);
}

std::vector<TAMMParser::Symmetry_groupContext *> TAMMParser::Permut_symmetryContext::symmetry_group() {
  return getRuleContexts<TAMMParser::Symmetry_groupContext>();
}

TAMMParser::Symmetry_groupContext* TAMMParser::Permut_symmetryContext::symmetry_group(size_t i) {
  return getRuleContext<TAMMParser::Symmetry_groupContext>(i);
}


size_t TAMMParser::Permut_symmetryContext::getRuleIndex() const {
  return TAMMParser::RulePermut_symmetry;
}

antlrcpp::Any TAMMParser::Permut_symmetryContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitPermut_symmetry(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Permut_symmetryContext* TAMMParser::permut_symmetry() {
  Permut_symmetryContext *_localctx = _tracker.createInstance<Permut_symmetryContext>(_ctx, getState());
  enterRule(_localctx, 32, TAMMParser::RulePermut_symmetry);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(154);
    match(TAMMParser::COLON);
    setState(156); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(155);
      symmetry_group();
      setState(158); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == TAMMParser::LPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Symmetry_groupContext ------------------------------------------------------------------

TAMMParser::Symmetry_groupContext::Symmetry_groupContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Symmetry_groupContext::LPAREN() {
  return getToken(TAMMParser::LPAREN, 0);
}

TAMMParser::Num_listContext* TAMMParser::Symmetry_groupContext::num_list() {
  return getRuleContext<TAMMParser::Num_listContext>(0);
}

tree::TerminalNode* TAMMParser::Symmetry_groupContext::RPAREN() {
  return getToken(TAMMParser::RPAREN, 0);
}


size_t TAMMParser::Symmetry_groupContext::getRuleIndex() const {
  return TAMMParser::RuleSymmetry_group;
}

antlrcpp::Any TAMMParser::Symmetry_groupContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitSymmetry_group(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Symmetry_groupContext* TAMMParser::symmetry_group() {
  Symmetry_groupContext *_localctx = _tracker.createInstance<Symmetry_groupContext>(_ctx, getState());
  enterRule(_localctx, 34, TAMMParser::RuleSymmetry_group);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(160);
    match(TAMMParser::LPAREN);
    setState(161);
    num_list();
    setState(162);
    match(TAMMParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Expansion_declarationContext ------------------------------------------------------------------

TAMMParser::Expansion_declarationContext::Expansion_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Expansion_declarationContext::EXPAND() {
  return getToken(TAMMParser::EXPAND, 0);
}

TAMMParser::Id_listContext* TAMMParser::Expansion_declarationContext::id_list() {
  return getRuleContext<TAMMParser::Id_listContext>(0);
}

tree::TerminalNode* TAMMParser::Expansion_declarationContext::SEMI() {
  return getToken(TAMMParser::SEMI, 0);
}


size_t TAMMParser::Expansion_declarationContext::getRuleIndex() const {
  return TAMMParser::RuleExpansion_declaration;
}

antlrcpp::Any TAMMParser::Expansion_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitExpansion_declaration(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Expansion_declarationContext* TAMMParser::expansion_declaration() {
  Expansion_declarationContext *_localctx = _tracker.createInstance<Expansion_declarationContext>(_ctx, getState());
  enterRule(_localctx, 36, TAMMParser::RuleExpansion_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(164);
    match(TAMMParser::EXPAND);
    setState(165);
    id_list();
    setState(166);
    match(TAMMParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Volatile_declarationContext ------------------------------------------------------------------

TAMMParser::Volatile_declarationContext::Volatile_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Volatile_declarationContext::VOLATILE() {
  return getToken(TAMMParser::VOLATILE, 0);
}

TAMMParser::Id_listContext* TAMMParser::Volatile_declarationContext::id_list() {
  return getRuleContext<TAMMParser::Id_listContext>(0);
}

tree::TerminalNode* TAMMParser::Volatile_declarationContext::SEMI() {
  return getToken(TAMMParser::SEMI, 0);
}


size_t TAMMParser::Volatile_declarationContext::getRuleIndex() const {
  return TAMMParser::RuleVolatile_declaration;
}

antlrcpp::Any TAMMParser::Volatile_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitVolatile_declaration(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Volatile_declarationContext* TAMMParser::volatile_declaration() {
  Volatile_declarationContext *_localctx = _tracker.createInstance<Volatile_declarationContext>(_ctx, getState());
  enterRule(_localctx, 38, TAMMParser::RuleVolatile_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(168);
    match(TAMMParser::VOLATILE);
    setState(169);
    id_list();
    setState(170);
    match(TAMMParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Iteration_declarationContext ------------------------------------------------------------------

TAMMParser::Iteration_declarationContext::Iteration_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Iteration_declarationContext::ITERATION() {
  return getToken(TAMMParser::ITERATION, 0);
}

tree::TerminalNode* TAMMParser::Iteration_declarationContext::EQUALS() {
  return getToken(TAMMParser::EQUALS, 0);
}

TAMMParser::Numerical_constantContext* TAMMParser::Iteration_declarationContext::numerical_constant() {
  return getRuleContext<TAMMParser::Numerical_constantContext>(0);
}

tree::TerminalNode* TAMMParser::Iteration_declarationContext::SEMI() {
  return getToken(TAMMParser::SEMI, 0);
}


size_t TAMMParser::Iteration_declarationContext::getRuleIndex() const {
  return TAMMParser::RuleIteration_declaration;
}

antlrcpp::Any TAMMParser::Iteration_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitIteration_declaration(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Iteration_declarationContext* TAMMParser::iteration_declaration() {
  Iteration_declarationContext *_localctx = _tracker.createInstance<Iteration_declarationContext>(_ctx, getState());
  enterRule(_localctx, 40, TAMMParser::RuleIteration_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(172);
    match(TAMMParser::ITERATION);
    setState(173);
    match(TAMMParser::EQUALS);
    setState(174);
    numerical_constant();
    setState(175);
    match(TAMMParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

TAMMParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAMMParser::Assignment_statementContext* TAMMParser::StatementContext::assignment_statement() {
  return getRuleContext<TAMMParser::Assignment_statementContext>(0);
}


size_t TAMMParser::StatementContext::getRuleIndex() const {
  return TAMMParser::RuleStatement;
}

antlrcpp::Any TAMMParser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::StatementContext* TAMMParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 42, TAMMParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(177);
    assignment_statement();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Assignment_statementContext ------------------------------------------------------------------

TAMMParser::Assignment_statementContext::Assignment_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAMMParser::ExpressionContext *> TAMMParser::Assignment_statementContext::expression() {
  return getRuleContexts<TAMMParser::ExpressionContext>();
}

TAMMParser::ExpressionContext* TAMMParser::Assignment_statementContext::expression(size_t i) {
  return getRuleContext<TAMMParser::ExpressionContext>(i);
}

TAMMParser::Assignment_operatorContext* TAMMParser::Assignment_statementContext::assignment_operator() {
  return getRuleContext<TAMMParser::Assignment_operatorContext>(0);
}

tree::TerminalNode* TAMMParser::Assignment_statementContext::SEMI() {
  return getToken(TAMMParser::SEMI, 0);
}

TAMMParser::IdentifierContext* TAMMParser::Assignment_statementContext::identifier() {
  return getRuleContext<TAMMParser::IdentifierContext>(0);
}

tree::TerminalNode* TAMMParser::Assignment_statementContext::COLON() {
  return getToken(TAMMParser::COLON, 0);
}


size_t TAMMParser::Assignment_statementContext::getRuleIndex() const {
  return TAMMParser::RuleAssignment_statement;
}

antlrcpp::Any TAMMParser::Assignment_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitAssignment_statement(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Assignment_statementContext* TAMMParser::assignment_statement() {
  Assignment_statementContext *_localctx = _tracker.createInstance<Assignment_statementContext>(_ctx, getState());
  enterRule(_localctx, 44, TAMMParser::RuleAssignment_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(182);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx)) {
    case 1: {
      setState(179);
      identifier();
      setState(180);
      match(TAMMParser::COLON);
      break;
    }

    }
    setState(184);
    expression();
    setState(185);
    assignment_operator();
    setState(186);
    expression();
    setState(187);
    match(TAMMParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Assignment_operatorContext ------------------------------------------------------------------

TAMMParser::Assignment_operatorContext::Assignment_operatorContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Assignment_operatorContext::EQUALS() {
  return getToken(TAMMParser::EQUALS, 0);
}

tree::TerminalNode* TAMMParser::Assignment_operatorContext::TIMESEQUAL() {
  return getToken(TAMMParser::TIMESEQUAL, 0);
}

tree::TerminalNode* TAMMParser::Assignment_operatorContext::PLUSEQUAL() {
  return getToken(TAMMParser::PLUSEQUAL, 0);
}

tree::TerminalNode* TAMMParser::Assignment_operatorContext::MINUSEQUAL() {
  return getToken(TAMMParser::MINUSEQUAL, 0);
}


size_t TAMMParser::Assignment_operatorContext::getRuleIndex() const {
  return TAMMParser::RuleAssignment_operator;
}

antlrcpp::Any TAMMParser::Assignment_operatorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitAssignment_operator(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Assignment_operatorContext* TAMMParser::assignment_operator() {
  Assignment_operatorContext *_localctx = _tracker.createInstance<Assignment_operatorContext>(_ctx, getState());
  enterRule(_localctx, 46, TAMMParser::RuleAssignment_operator);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(189);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << TAMMParser::EQUALS)
      | (1ULL << TAMMParser::TIMESEQUAL)
      | (1ULL << TAMMParser::PLUSEQUAL)
      | (1ULL << TAMMParser::MINUSEQUAL))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Unary_expressionContext ------------------------------------------------------------------

TAMMParser::Unary_expressionContext::Unary_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAMMParser::Primary_expressionContext* TAMMParser::Unary_expressionContext::primary_expression() {
  return getRuleContext<TAMMParser::Primary_expressionContext>(0);
}

tree::TerminalNode* TAMMParser::Unary_expressionContext::PLUS() {
  return getToken(TAMMParser::PLUS, 0);
}

TAMMParser::Unary_expressionContext* TAMMParser::Unary_expressionContext::unary_expression() {
  return getRuleContext<TAMMParser::Unary_expressionContext>(0);
}

tree::TerminalNode* TAMMParser::Unary_expressionContext::MINUS() {
  return getToken(TAMMParser::MINUS, 0);
}


size_t TAMMParser::Unary_expressionContext::getRuleIndex() const {
  return TAMMParser::RuleUnary_expression;
}

antlrcpp::Any TAMMParser::Unary_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitUnary_expression(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Unary_expressionContext* TAMMParser::unary_expression() {
  Unary_expressionContext *_localctx = _tracker.createInstance<Unary_expressionContext>(_ctx, getState());
  enterRule(_localctx, 48, TAMMParser::RuleUnary_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(196);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAMMParser::LPAREN:
      case TAMMParser::ID:
      case TAMMParser::ICONST:
      case TAMMParser::FRAC:
      case TAMMParser::FCONST: {
        enterOuterAlt(_localctx, 1);
        setState(191);
        primary_expression();
        break;
      }

      case TAMMParser::PLUS: {
        enterOuterAlt(_localctx, 2);
        setState(192);
        match(TAMMParser::PLUS);
        setState(193);
        unary_expression();
        break;
      }

      case TAMMParser::MINUS: {
        enterOuterAlt(_localctx, 3);
        setState(194);
        match(TAMMParser::MINUS);
        setState(195);
        unary_expression();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Primary_expressionContext ------------------------------------------------------------------

TAMMParser::Primary_expressionContext::Primary_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

TAMMParser::Numerical_constantContext* TAMMParser::Primary_expressionContext::numerical_constant() {
  return getRuleContext<TAMMParser::Numerical_constantContext>(0);
}

TAMMParser::Array_referenceContext* TAMMParser::Primary_expressionContext::array_reference() {
  return getRuleContext<TAMMParser::Array_referenceContext>(0);
}

tree::TerminalNode* TAMMParser::Primary_expressionContext::LPAREN() {
  return getToken(TAMMParser::LPAREN, 0);
}

TAMMParser::ExpressionContext* TAMMParser::Primary_expressionContext::expression() {
  return getRuleContext<TAMMParser::ExpressionContext>(0);
}

tree::TerminalNode* TAMMParser::Primary_expressionContext::RPAREN() {
  return getToken(TAMMParser::RPAREN, 0);
}


size_t TAMMParser::Primary_expressionContext::getRuleIndex() const {
  return TAMMParser::RulePrimary_expression;
}

antlrcpp::Any TAMMParser::Primary_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitPrimary_expression(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Primary_expressionContext* TAMMParser::primary_expression() {
  Primary_expressionContext *_localctx = _tracker.createInstance<Primary_expressionContext>(_ctx, getState());
  enterRule(_localctx, 50, TAMMParser::RulePrimary_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(204);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case TAMMParser::ICONST:
      case TAMMParser::FRAC:
      case TAMMParser::FCONST: {
        enterOuterAlt(_localctx, 1);
        setState(198);
        numerical_constant();
        break;
      }

      case TAMMParser::ID: {
        enterOuterAlt(_localctx, 2);
        setState(199);
        array_reference();
        break;
      }

      case TAMMParser::LPAREN: {
        enterOuterAlt(_localctx, 3);
        setState(200);
        match(TAMMParser::LPAREN);
        setState(201);
        expression();
        setState(202);
        match(TAMMParser::RPAREN);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Array_referenceContext ------------------------------------------------------------------

TAMMParser::Array_referenceContext::Array_referenceContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::Array_referenceContext::ID() {
  return getToken(TAMMParser::ID, 0);
}

tree::TerminalNode* TAMMParser::Array_referenceContext::LBRACKET() {
  return getToken(TAMMParser::LBRACKET, 0);
}

TAMMParser::Id_list_optContext* TAMMParser::Array_referenceContext::id_list_opt() {
  return getRuleContext<TAMMParser::Id_list_optContext>(0);
}

tree::TerminalNode* TAMMParser::Array_referenceContext::RBRACKET() {
  return getToken(TAMMParser::RBRACKET, 0);
}


size_t TAMMParser::Array_referenceContext::getRuleIndex() const {
  return TAMMParser::RuleArray_reference;
}

antlrcpp::Any TAMMParser::Array_referenceContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitArray_reference(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Array_referenceContext* TAMMParser::array_reference() {
  Array_referenceContext *_localctx = _tracker.createInstance<Array_referenceContext>(_ctx, getState());
  enterRule(_localctx, 52, TAMMParser::RuleArray_reference);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(206);
    match(TAMMParser::ID);
    setState(211);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == TAMMParser::LBRACKET) {
      setState(207);
      match(TAMMParser::LBRACKET);
      setState(208);
      id_list_opt();
      setState(209);
      match(TAMMParser::RBRACKET);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PlusORminusContext ------------------------------------------------------------------

TAMMParser::PlusORminusContext::PlusORminusContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TAMMParser::PlusORminusContext::PLUS() {
  return getToken(TAMMParser::PLUS, 0);
}

tree::TerminalNode* TAMMParser::PlusORminusContext::MINUS() {
  return getToken(TAMMParser::MINUS, 0);
}


size_t TAMMParser::PlusORminusContext::getRuleIndex() const {
  return TAMMParser::RulePlusORminus;
}

antlrcpp::Any TAMMParser::PlusORminusContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitPlusORminus(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::PlusORminusContext* TAMMParser::plusORminus() {
  PlusORminusContext *_localctx = _tracker.createInstance<PlusORminusContext>(_ctx, getState());
  enterRule(_localctx, 54, TAMMParser::RulePlusORminus);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(213);
    _la = _input->LA(1);
    if (!(_la == TAMMParser::PLUS

    || _la == TAMMParser::MINUS)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

TAMMParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAMMParser::Multiplicative_expressionContext *> TAMMParser::ExpressionContext::multiplicative_expression() {
  return getRuleContexts<TAMMParser::Multiplicative_expressionContext>();
}

TAMMParser::Multiplicative_expressionContext* TAMMParser::ExpressionContext::multiplicative_expression(size_t i) {
  return getRuleContext<TAMMParser::Multiplicative_expressionContext>(i);
}

std::vector<TAMMParser::PlusORminusContext *> TAMMParser::ExpressionContext::plusORminus() {
  return getRuleContexts<TAMMParser::PlusORminusContext>();
}

TAMMParser::PlusORminusContext* TAMMParser::ExpressionContext::plusORminus(size_t i) {
  return getRuleContext<TAMMParser::PlusORminusContext>(i);
}


size_t TAMMParser::ExpressionContext::getRuleIndex() const {
  return TAMMParser::RuleExpression;
}

antlrcpp::Any TAMMParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::ExpressionContext* TAMMParser::expression() {
  ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, getState());
  enterRule(_localctx, 56, TAMMParser::RuleExpression);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(215);
    multiplicative_expression();
    setState(221);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TAMMParser::PLUS

    || _la == TAMMParser::MINUS) {
      setState(216);
      plusORminus();
      setState(217);
      multiplicative_expression();
      setState(223);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Multiplicative_expressionContext ------------------------------------------------------------------

TAMMParser::Multiplicative_expressionContext::Multiplicative_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<TAMMParser::Unary_expressionContext *> TAMMParser::Multiplicative_expressionContext::unary_expression() {
  return getRuleContexts<TAMMParser::Unary_expressionContext>();
}

TAMMParser::Unary_expressionContext* TAMMParser::Multiplicative_expressionContext::unary_expression(size_t i) {
  return getRuleContext<TAMMParser::Unary_expressionContext>(i);
}

std::vector<tree::TerminalNode *> TAMMParser::Multiplicative_expressionContext::TIMES() {
  return getTokens(TAMMParser::TIMES);
}

tree::TerminalNode* TAMMParser::Multiplicative_expressionContext::TIMES(size_t i) {
  return getToken(TAMMParser::TIMES, i);
}


size_t TAMMParser::Multiplicative_expressionContext::getRuleIndex() const {
  return TAMMParser::RuleMultiplicative_expression;
}

antlrcpp::Any TAMMParser::Multiplicative_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TAMMVisitor*>(visitor))
    return parserVisitor->visitMultiplicative_expression(this);
  else
    return visitor->visitChildren(this);
}

TAMMParser::Multiplicative_expressionContext* TAMMParser::multiplicative_expression() {
  Multiplicative_expressionContext *_localctx = _tracker.createInstance<Multiplicative_expressionContext>(_ctx, getState());
  enterRule(_localctx, 58, TAMMParser::RuleMultiplicative_expression);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(224);
    unary_expression();
    setState(229);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == TAMMParser::TIMES) {
      setState(225);
      match(TAMMParser::TIMES);
      setState(226);
      unary_expression();
      setState(231);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

// Static vars and initialization.
std::vector<dfa::DFA> TAMMParser::_decisionToDFA;
atn::PredictionContextCache TAMMParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN TAMMParser::_atn;
std::vector<uint16_t> TAMMParser::_serializedATN;

std::vector<std::string> TAMMParser::_ruleNames = {
  "translation_unit", "compound_element_list", "element_list", "compound_element", 
  "element", "declaration", "id_list_opt", "id_list", "num_list", "identifier", 
  "numerical_constant", "range_declaration", "index_declaration", "array_declaration", 
  "array_structure", "array_structure_list", "permut_symmetry", "symmetry_group", 
  "expansion_declaration", "volatile_declaration", "iteration_declaration", 
  "statement", "assignment_statement", "assignment_operator", "unary_expression", 
  "primary_expression", "array_reference", "plusORminus", "expression", 
  "multiplicative_expression"
};

std::vector<std::string> TAMMParser::_literalNames = {
  "", "'range'", "'index'", "'array'", "'expand'", "'volatile'", "'iteration'", 
  "'+'", "'-'", "'*'", "'='", "'*='", "'+='", "'-='", "'('", "')'", "'{'", 
  "'}'", "'['", "']'", "','", "':'", "';'"
};

std::vector<std::string> TAMMParser::_symbolicNames = {
  "", "RANGE", "INDEX", "ARRAY", "EXPAND", "VOLATILE", "ITERATION", "PLUS", 
  "MINUS", "TIMES", "EQUALS", "TIMESEQUAL", "PLUSEQUAL", "MINUSEQUAL", "LPAREN", 
  "RPAREN", "LBRACE", "RBRACE", "LBRACKET", "RBRACKET", "COMMA", "COLON", 
  "SEMI", "ID", "ICONST", "FRAC", "FCONST", "Whitespace", "Newline", "BlockComment", 
  "LineComment"
};

dfa::Vocabulary TAMMParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> TAMMParser::_tokenNames;

TAMMParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x430, 0xd6d1, 0x8206, 0xad2d, 0x4417, 0xaef1, 0x8d80, 0xaadd, 
    0x3, 0x20, 0xeb, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 0x9, 
    0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 0x4, 
    0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 0x9, 
    0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 0x4, 
    0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 0x12, 
    0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 0x9, 
    0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x4, 0x18, 0x9, 0x18, 
    0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 0x9, 0x1a, 0x4, 0x1b, 0x9, 0x1b, 0x4, 
    0x1c, 0x9, 0x1c, 0x4, 0x1d, 0x9, 0x1d, 0x4, 0x1e, 0x9, 0x1e, 0x4, 0x1f, 
    0x9, 0x1f, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x3, 0x7, 0x3, 0x43, 0xa, 
    0x3, 0xc, 0x3, 0xe, 0x3, 0x46, 0xb, 0x3, 0x3, 0x4, 0x7, 0x4, 0x49, 0xa, 
    0x4, 0xc, 0x4, 0xe, 0x4, 0x4c, 0xb, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 
    0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x55, 0xa, 0x6, 0x3, 
    0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0x5d, 
    0xa, 0x7, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x61, 0xa, 0x8, 0x3, 0x9, 0x3, 
    0x9, 0x3, 0x9, 0x7, 0x9, 0x66, 0xa, 0x9, 0xc, 0x9, 0xe, 0x9, 0x69, 0xb, 
    0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x7, 0xa, 0x6e, 0xa, 0xa, 0xc, 0xa, 
    0xe, 0xa, 0x71, 0xb, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 
    0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 0x3, 
    0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0xf, 0x3, 0xf, 0x5, 0xf, 0x87, 0xa, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 
    0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 
    0x10, 0x5, 0x10, 0x93, 0xa, 0x10, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x7, 
    0x11, 0x98, 0xa, 0x11, 0xc, 0x11, 0xe, 0x11, 0x9b, 0xb, 0x11, 0x3, 0x12, 
    0x3, 0x12, 0x6, 0x12, 0x9f, 0xa, 0x12, 0xd, 0x12, 0xe, 0x12, 0xa0, 0x3, 
    0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x16, 0x3, 
    0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x18, 
    0x3, 0x18, 0x3, 0x18, 0x5, 0x18, 0xb9, 0xa, 0x18, 0x3, 0x18, 0x3, 0x18, 
    0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x1a, 0x3, 
    0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x5, 0x1a, 0xc7, 0xa, 0x1a, 0x3, 
    0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x5, 0x1b, 
    0xcf, 0xa, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 
    0x5, 0x1c, 0xd6, 0xa, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 
    0x3, 0x1e, 0x3, 0x1e, 0x7, 0x1e, 0xde, 0xa, 0x1e, 0xc, 0x1e, 0xe, 0x1e, 
    0xe1, 0xb, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x7, 0x1f, 0xe6, 0xa, 
    0x1f, 0xc, 0x1f, 0xe, 0x1f, 0xe9, 0xb, 0x1f, 0x3, 0x1f, 0x2, 0x2, 0x20, 
    0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 
    0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 
    0x34, 0x36, 0x38, 0x3a, 0x3c, 0x2, 0x5, 0x3, 0x2, 0x1a, 0x1c, 0x3, 0x2, 
    0xc, 0xf, 0x3, 0x2, 0x9, 0xa, 0xe3, 0x2, 0x3e, 0x3, 0x2, 0x2, 0x2, 0x4, 
    0x44, 0x3, 0x2, 0x2, 0x2, 0x6, 0x4a, 0x3, 0x2, 0x2, 0x2, 0x8, 0x4d, 
    0x3, 0x2, 0x2, 0x2, 0xa, 0x54, 0x3, 0x2, 0x2, 0x2, 0xc, 0x5c, 0x3, 0x2, 
    0x2, 0x2, 0xe, 0x60, 0x3, 0x2, 0x2, 0x2, 0x10, 0x62, 0x3, 0x2, 0x2, 
    0x2, 0x12, 0x6a, 0x3, 0x2, 0x2, 0x2, 0x14, 0x72, 0x3, 0x2, 0x2, 0x2, 
    0x16, 0x74, 0x3, 0x2, 0x2, 0x2, 0x18, 0x76, 0x3, 0x2, 0x2, 0x2, 0x1a, 
    0x7c, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x82, 0x3, 0x2, 0x2, 0x2, 0x1e, 0x8a, 
    0x3, 0x2, 0x2, 0x2, 0x20, 0x94, 0x3, 0x2, 0x2, 0x2, 0x22, 0x9c, 0x3, 
    0x2, 0x2, 0x2, 0x24, 0xa2, 0x3, 0x2, 0x2, 0x2, 0x26, 0xa6, 0x3, 0x2, 
    0x2, 0x2, 0x28, 0xaa, 0x3, 0x2, 0x2, 0x2, 0x2a, 0xae, 0x3, 0x2, 0x2, 
    0x2, 0x2c, 0xb3, 0x3, 0x2, 0x2, 0x2, 0x2e, 0xb8, 0x3, 0x2, 0x2, 0x2, 
    0x30, 0xbf, 0x3, 0x2, 0x2, 0x2, 0x32, 0xc6, 0x3, 0x2, 0x2, 0x2, 0x34, 
    0xce, 0x3, 0x2, 0x2, 0x2, 0x36, 0xd0, 0x3, 0x2, 0x2, 0x2, 0x38, 0xd7, 
    0x3, 0x2, 0x2, 0x2, 0x3a, 0xd9, 0x3, 0x2, 0x2, 0x2, 0x3c, 0xe2, 0x3, 
    0x2, 0x2, 0x2, 0x3e, 0x3f, 0x5, 0x4, 0x3, 0x2, 0x3f, 0x40, 0x7, 0x2, 
    0x2, 0x3, 0x40, 0x3, 0x3, 0x2, 0x2, 0x2, 0x41, 0x43, 0x5, 0x8, 0x5, 
    0x2, 0x42, 0x41, 0x3, 0x2, 0x2, 0x2, 0x43, 0x46, 0x3, 0x2, 0x2, 0x2, 
    0x44, 0x42, 0x3, 0x2, 0x2, 0x2, 0x44, 0x45, 0x3, 0x2, 0x2, 0x2, 0x45, 
    0x5, 0x3, 0x2, 0x2, 0x2, 0x46, 0x44, 0x3, 0x2, 0x2, 0x2, 0x47, 0x49, 
    0x5, 0xa, 0x6, 0x2, 0x48, 0x47, 0x3, 0x2, 0x2, 0x2, 0x49, 0x4c, 0x3, 
    0x2, 0x2, 0x2, 0x4a, 0x48, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x4b, 0x3, 0x2, 
    0x2, 0x2, 0x4b, 0x7, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x4a, 0x3, 0x2, 0x2, 
    0x2, 0x4d, 0x4e, 0x5, 0x14, 0xb, 0x2, 0x4e, 0x4f, 0x7, 0x12, 0x2, 0x2, 
    0x4f, 0x50, 0x5, 0x6, 0x4, 0x2, 0x50, 0x51, 0x7, 0x13, 0x2, 0x2, 0x51, 
    0x9, 0x3, 0x2, 0x2, 0x2, 0x52, 0x55, 0x5, 0xc, 0x7, 0x2, 0x53, 0x55, 
    0x5, 0x2c, 0x17, 0x2, 0x54, 0x52, 0x3, 0x2, 0x2, 0x2, 0x54, 0x53, 0x3, 
    0x2, 0x2, 0x2, 0x55, 0xb, 0x3, 0x2, 0x2, 0x2, 0x56, 0x5d, 0x5, 0x18, 
    0xd, 0x2, 0x57, 0x5d, 0x5, 0x1a, 0xe, 0x2, 0x58, 0x5d, 0x5, 0x1c, 0xf, 
    0x2, 0x59, 0x5d, 0x5, 0x26, 0x14, 0x2, 0x5a, 0x5d, 0x5, 0x28, 0x15, 
    0x2, 0x5b, 0x5d, 0x5, 0x2a, 0x16, 0x2, 0x5c, 0x56, 0x3, 0x2, 0x2, 0x2, 
    0x5c, 0x57, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x58, 0x3, 0x2, 0x2, 0x2, 0x5c, 
    0x59, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x5a, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x5b, 
    0x3, 0x2, 0x2, 0x2, 0x5d, 0xd, 0x3, 0x2, 0x2, 0x2, 0x5e, 0x61, 0x3, 
    0x2, 0x2, 0x2, 0x5f, 0x61, 0x5, 0x10, 0x9, 0x2, 0x60, 0x5e, 0x3, 0x2, 
    0x2, 0x2, 0x60, 0x5f, 0x3, 0x2, 0x2, 0x2, 0x61, 0xf, 0x3, 0x2, 0x2, 
    0x2, 0x62, 0x67, 0x5, 0x14, 0xb, 0x2, 0x63, 0x64, 0x7, 0x16, 0x2, 0x2, 
    0x64, 0x66, 0x5, 0x14, 0xb, 0x2, 0x65, 0x63, 0x3, 0x2, 0x2, 0x2, 0x66, 
    0x69, 0x3, 0x2, 0x2, 0x2, 0x67, 0x65, 0x3, 0x2, 0x2, 0x2, 0x67, 0x68, 
    0x3, 0x2, 0x2, 0x2, 0x68, 0x11, 0x3, 0x2, 0x2, 0x2, 0x69, 0x67, 0x3, 
    0x2, 0x2, 0x2, 0x6a, 0x6f, 0x5, 0x16, 0xc, 0x2, 0x6b, 0x6c, 0x7, 0x16, 
    0x2, 0x2, 0x6c, 0x6e, 0x5, 0x16, 0xc, 0x2, 0x6d, 0x6b, 0x3, 0x2, 0x2, 
    0x2, 0x6e, 0x71, 0x3, 0x2, 0x2, 0x2, 0x6f, 0x6d, 0x3, 0x2, 0x2, 0x2, 
    0x6f, 0x70, 0x3, 0x2, 0x2, 0x2, 0x70, 0x13, 0x3, 0x2, 0x2, 0x2, 0x71, 
    0x6f, 0x3, 0x2, 0x2, 0x2, 0x72, 0x73, 0x7, 0x19, 0x2, 0x2, 0x73, 0x15, 
    0x3, 0x2, 0x2, 0x2, 0x74, 0x75, 0x9, 0x2, 0x2, 0x2, 0x75, 0x17, 0x3, 
    0x2, 0x2, 0x2, 0x76, 0x77, 0x7, 0x3, 0x2, 0x2, 0x77, 0x78, 0x5, 0x10, 
    0x9, 0x2, 0x78, 0x79, 0x7, 0xc, 0x2, 0x2, 0x79, 0x7a, 0x5, 0x16, 0xc, 
    0x2, 0x7a, 0x7b, 0x7, 0x18, 0x2, 0x2, 0x7b, 0x19, 0x3, 0x2, 0x2, 0x2, 
    0x7c, 0x7d, 0x7, 0x4, 0x2, 0x2, 0x7d, 0x7e, 0x5, 0x10, 0x9, 0x2, 0x7e, 
    0x7f, 0x7, 0xc, 0x2, 0x2, 0x7f, 0x80, 0x5, 0x14, 0xb, 0x2, 0x80, 0x81, 
    0x7, 0x18, 0x2, 0x2, 0x81, 0x1b, 0x3, 0x2, 0x2, 0x2, 0x82, 0x83, 0x7, 
    0x5, 0x2, 0x2, 0x83, 0x86, 0x5, 0x20, 0x11, 0x2, 0x84, 0x85, 0x7, 0x17, 
    0x2, 0x2, 0x85, 0x87, 0x5, 0x14, 0xb, 0x2, 0x86, 0x84, 0x3, 0x2, 0x2, 
    0x2, 0x86, 0x87, 0x3, 0x2, 0x2, 0x2, 0x87, 0x88, 0x3, 0x2, 0x2, 0x2, 
    0x88, 0x89, 0x7, 0x18, 0x2, 0x2, 0x89, 0x1d, 0x3, 0x2, 0x2, 0x2, 0x8a, 
    0x8b, 0x7, 0x19, 0x2, 0x2, 0x8b, 0x8c, 0x7, 0x14, 0x2, 0x2, 0x8c, 0x8d, 
    0x5, 0xe, 0x8, 0x2, 0x8d, 0x8e, 0x7, 0x15, 0x2, 0x2, 0x8e, 0x8f, 0x7, 
    0x14, 0x2, 0x2, 0x8f, 0x90, 0x5, 0xe, 0x8, 0x2, 0x90, 0x92, 0x7, 0x15, 
    0x2, 0x2, 0x91, 0x93, 0x5, 0x22, 0x12, 0x2, 0x92, 0x91, 0x3, 0x2, 0x2, 
    0x2, 0x92, 0x93, 0x3, 0x2, 0x2, 0x2, 0x93, 0x1f, 0x3, 0x2, 0x2, 0x2, 
    0x94, 0x99, 0x5, 0x1e, 0x10, 0x2, 0x95, 0x96, 0x7, 0x16, 0x2, 0x2, 0x96, 
    0x98, 0x5, 0x1e, 0x10, 0x2, 0x97, 0x95, 0x3, 0x2, 0x2, 0x2, 0x98, 0x9b, 
    0x3, 0x2, 0x2, 0x2, 0x99, 0x97, 0x3, 0x2, 0x2, 0x2, 0x99, 0x9a, 0x3, 
    0x2, 0x2, 0x2, 0x9a, 0x21, 0x3, 0x2, 0x2, 0x2, 0x9b, 0x99, 0x3, 0x2, 
    0x2, 0x2, 0x9c, 0x9e, 0x7, 0x17, 0x2, 0x2, 0x9d, 0x9f, 0x5, 0x24, 0x13, 
    0x2, 0x9e, 0x9d, 0x3, 0x2, 0x2, 0x2, 0x9f, 0xa0, 0x3, 0x2, 0x2, 0x2, 
    0xa0, 0x9e, 0x3, 0x2, 0x2, 0x2, 0xa0, 0xa1, 0x3, 0x2, 0x2, 0x2, 0xa1, 
    0x23, 0x3, 0x2, 0x2, 0x2, 0xa2, 0xa3, 0x7, 0x10, 0x2, 0x2, 0xa3, 0xa4, 
    0x5, 0x12, 0xa, 0x2, 0xa4, 0xa5, 0x7, 0x11, 0x2, 0x2, 0xa5, 0x25, 0x3, 
    0x2, 0x2, 0x2, 0xa6, 0xa7, 0x7, 0x6, 0x2, 0x2, 0xa7, 0xa8, 0x5, 0x10, 
    0x9, 0x2, 0xa8, 0xa9, 0x7, 0x18, 0x2, 0x2, 0xa9, 0x27, 0x3, 0x2, 0x2, 
    0x2, 0xaa, 0xab, 0x7, 0x7, 0x2, 0x2, 0xab, 0xac, 0x5, 0x10, 0x9, 0x2, 
    0xac, 0xad, 0x7, 0x18, 0x2, 0x2, 0xad, 0x29, 0x3, 0x2, 0x2, 0x2, 0xae, 
    0xaf, 0x7, 0x8, 0x2, 0x2, 0xaf, 0xb0, 0x7, 0xc, 0x2, 0x2, 0xb0, 0xb1, 
    0x5, 0x16, 0xc, 0x2, 0xb1, 0xb2, 0x7, 0x18, 0x2, 0x2, 0xb2, 0x2b, 0x3, 
    0x2, 0x2, 0x2, 0xb3, 0xb4, 0x5, 0x2e, 0x18, 0x2, 0xb4, 0x2d, 0x3, 0x2, 
    0x2, 0x2, 0xb5, 0xb6, 0x5, 0x14, 0xb, 0x2, 0xb6, 0xb7, 0x7, 0x17, 0x2, 
    0x2, 0xb7, 0xb9, 0x3, 0x2, 0x2, 0x2, 0xb8, 0xb5, 0x3, 0x2, 0x2, 0x2, 
    0xb8, 0xb9, 0x3, 0x2, 0x2, 0x2, 0xb9, 0xba, 0x3, 0x2, 0x2, 0x2, 0xba, 
    0xbb, 0x5, 0x3a, 0x1e, 0x2, 0xbb, 0xbc, 0x5, 0x30, 0x19, 0x2, 0xbc, 
    0xbd, 0x5, 0x3a, 0x1e, 0x2, 0xbd, 0xbe, 0x7, 0x18, 0x2, 0x2, 0xbe, 0x2f, 
    0x3, 0x2, 0x2, 0x2, 0xbf, 0xc0, 0x9, 0x3, 0x2, 0x2, 0xc0, 0x31, 0x3, 
    0x2, 0x2, 0x2, 0xc1, 0xc7, 0x5, 0x34, 0x1b, 0x2, 0xc2, 0xc3, 0x7, 0x9, 
    0x2, 0x2, 0xc3, 0xc7, 0x5, 0x32, 0x1a, 0x2, 0xc4, 0xc5, 0x7, 0xa, 0x2, 
    0x2, 0xc5, 0xc7, 0x5, 0x32, 0x1a, 0x2, 0xc6, 0xc1, 0x3, 0x2, 0x2, 0x2, 
    0xc6, 0xc2, 0x3, 0x2, 0x2, 0x2, 0xc6, 0xc4, 0x3, 0x2, 0x2, 0x2, 0xc7, 
    0x33, 0x3, 0x2, 0x2, 0x2, 0xc8, 0xcf, 0x5, 0x16, 0xc, 0x2, 0xc9, 0xcf, 
    0x5, 0x36, 0x1c, 0x2, 0xca, 0xcb, 0x7, 0x10, 0x2, 0x2, 0xcb, 0xcc, 0x5, 
    0x3a, 0x1e, 0x2, 0xcc, 0xcd, 0x7, 0x11, 0x2, 0x2, 0xcd, 0xcf, 0x3, 0x2, 
    0x2, 0x2, 0xce, 0xc8, 0x3, 0x2, 0x2, 0x2, 0xce, 0xc9, 0x3, 0x2, 0x2, 
    0x2, 0xce, 0xca, 0x3, 0x2, 0x2, 0x2, 0xcf, 0x35, 0x3, 0x2, 0x2, 0x2, 
    0xd0, 0xd5, 0x7, 0x19, 0x2, 0x2, 0xd1, 0xd2, 0x7, 0x14, 0x2, 0x2, 0xd2, 
    0xd3, 0x5, 0xe, 0x8, 0x2, 0xd3, 0xd4, 0x7, 0x15, 0x2, 0x2, 0xd4, 0xd6, 
    0x3, 0x2, 0x2, 0x2, 0xd5, 0xd1, 0x3, 0x2, 0x2, 0x2, 0xd5, 0xd6, 0x3, 
    0x2, 0x2, 0x2, 0xd6, 0x37, 0x3, 0x2, 0x2, 0x2, 0xd7, 0xd8, 0x9, 0x4, 
    0x2, 0x2, 0xd8, 0x39, 0x3, 0x2, 0x2, 0x2, 0xd9, 0xdf, 0x5, 0x3c, 0x1f, 
    0x2, 0xda, 0xdb, 0x5, 0x38, 0x1d, 0x2, 0xdb, 0xdc, 0x5, 0x3c, 0x1f, 
    0x2, 0xdc, 0xde, 0x3, 0x2, 0x2, 0x2, 0xdd, 0xda, 0x3, 0x2, 0x2, 0x2, 
    0xde, 0xe1, 0x3, 0x2, 0x2, 0x2, 0xdf, 0xdd, 0x3, 0x2, 0x2, 0x2, 0xdf, 
    0xe0, 0x3, 0x2, 0x2, 0x2, 0xe0, 0x3b, 0x3, 0x2, 0x2, 0x2, 0xe1, 0xdf, 
    0x3, 0x2, 0x2, 0x2, 0xe2, 0xe7, 0x5, 0x32, 0x1a, 0x2, 0xe3, 0xe4, 0x7, 
    0xb, 0x2, 0x2, 0xe4, 0xe6, 0x5, 0x32, 0x1a, 0x2, 0xe5, 0xe3, 0x3, 0x2, 
    0x2, 0x2, 0xe6, 0xe9, 0x3, 0x2, 0x2, 0x2, 0xe7, 0xe5, 0x3, 0x2, 0x2, 
    0x2, 0xe7, 0xe8, 0x3, 0x2, 0x2, 0x2, 0xe8, 0x3d, 0x3, 0x2, 0x2, 0x2, 
    0xe9, 0xe7, 0x3, 0x2, 0x2, 0x2, 0x13, 0x44, 0x4a, 0x54, 0x5c, 0x60, 
    0x67, 0x6f, 0x86, 0x92, 0x99, 0xa0, 0xb8, 0xc6, 0xce, 0xd5, 0xdf, 0xe7, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

TAMMParser::Initializer TAMMParser::_init;

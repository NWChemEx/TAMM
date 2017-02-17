
// Generated from TAMM.g4 by ANTLR 4.6

#pragma once


#include "antlr4-runtime.h"




class  TAMMLexer : public antlr4::Lexer {
public:
  enum {
    RANGE = 1, INDEX = 2, ARRAY = 3, SCALAR = 4, EXPAND = 5, VOLATILE = 6, 
    ITERATION = 7, PLUS = 8, MINUS = 9, TIMES = 10, EQUALS = 11, TIMESEQUAL = 12, 
    PLUSEQUAL = 13, MINUSEQUAL = 14, LPAREN = 15, RPAREN = 16, LBRACE = 17, 
    RBRACE = 18, LBRACKET = 19, RBRACKET = 20, COMMA = 21, COLON = 22, SEMI = 23, 
    ID = 24, ICONST = 25, FRAC = 26, FCONST = 27, Whitespace = 28, Newline = 29, 
    BlockComment = 30, LineComment = 31
  };

  TAMMLexer(antlr4::CharStream *input);
  ~TAMMLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string>& getRuleNames() const override;

  virtual const std::vector<std::string>& getModeNames() const override;
  virtual const std::vector<std::string>& getTokenNames() const override; // deprecated, use vocabulary instead
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;

  virtual const std::vector<uint16_t> getSerializedATN() const override;
  virtual const antlr4::atn::ATN& getATN() const override;

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;
  static std::vector<std::string> _modeNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};


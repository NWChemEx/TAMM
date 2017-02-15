
// Generated from TAMM.g4 by ANTLR 4.6

#pragma once


#include "antlr4-runtime.h"




class  TAMMLexer : public antlr4::Lexer {
public:
  enum {
    RANGE = 1, INDEX = 2, ARRAY = 3, EXPAND = 4, VOLATILE = 5, ITERATION = 6, 
    PLUS = 7, MINUS = 8, TIMES = 9, EQUALS = 10, TIMESEQUAL = 11, PLUSEQUAL = 12, 
    MINUSEQUAL = 13, LPAREN = 14, RPAREN = 15, LBRACE = 16, RBRACE = 17, 
    LBRACKET = 18, RBRACKET = 19, COMMA = 20, COLON = 21, SEMI = 22, ID = 23, 
    ICONST = 24, FRAC = 25, FCONST = 26, Whitespace = 27, Newline = 28, 
    BlockComment = 29, LineComment = 30
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


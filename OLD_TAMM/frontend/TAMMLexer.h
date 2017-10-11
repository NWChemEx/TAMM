
// Generated from TAMM.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"




class  TAMMLexer : public antlr4::Lexer {
public:
  enum {
    RANGE = 1, INDEX = 2, ARRAY = 3, SCALAR = 4, AUXBASIS = 5, PLUS = 6, 
    MINUS = 7, TIMES = 8, EQUALS = 9, TIMESEQUAL = 10, PLUSEQUAL = 11, MINUSEQUAL = 12, 
    LPAREN = 13, RPAREN = 14, LBRACE = 15, RBRACE = 16, LBRACKET = 17, RBRACKET = 18, 
    COMMA = 19, COLON = 20, SEMI = 21, ID = 22, ICONST = 23, FRAC = 24, 
    FCONST = 25, Whitespace = 26, Newline = 27, BlockComment = 28, LineComment = 29
  };

  TAMMLexer(antlr4::CharStream *input);
  ~TAMMLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string>& getRuleNames() const override;

  virtual const std::vector<std::string>& getChannelNames() const override;
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
  static std::vector<std::string> _channelNames;
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


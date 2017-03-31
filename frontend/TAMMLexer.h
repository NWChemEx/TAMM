
// Generated from TAMM.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"




class  TAMMLexer : public antlr4::Lexer {
public:
  enum {
    RANGE = 1, INDEX = 2, ARRAY = 3, SCALAR = 4, PLUS = 5, MINUS = 6, TIMES = 7, 
    EQUALS = 8, TIMESEQUAL = 9, PLUSEQUAL = 10, MINUSEQUAL = 11, LPAREN = 12, 
    RPAREN = 13, LBRACE = 14, RBRACE = 15, LBRACKET = 16, RBRACKET = 17, 
    COMMA = 18, COLON = 19, SEMI = 20, ID = 21, ICONST = 22, FRAC = 23, 
    FCONST = 24, Whitespace = 25, Newline = 26, BlockComment = 27, LineComment = 28
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


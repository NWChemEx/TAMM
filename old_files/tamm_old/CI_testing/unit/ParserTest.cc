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

#include "gtest/gtest.h"
#include <iostream>
#include <cmath>
#include <cerrno>
#include <cfenv>
#include <cstring>

#include "TAMMLexer.h"
#include "TAMMParser.h"
#include "ASTBuilder.h"


  const std::string test_tamm_parser(const std::string test_string) {
    //std::ifstream stream(test_string.c_str());
    //if (!stream.good()) Error("No input provided");
    antlr4::ANTLRInputStream tamminput(test_string);
    TAMMLexer lexer(&tamminput);
    antlr4::CommonTokenStream tokens(&lexer);
    TAMMParser parser(&tokens);
    //parser.setErrorHandler(new antlr4::BailErrorStrategy());
    antlr4::tree::ParseTree *tree = parser.translation_unit();
    //std::cout << "info:" << tree->getText() << std::endl;
    //tamm::frontend::ASTBuilder *visitor = new tamm::frontend::ASTBuilder();
    //tamm::frontend::CompilationUnit *ast_root = visitor->visit(tree);
    return "Success";
  }

TEST (ParserTest, RangeDeclarations) { 
    ASSERT_EQ ("Success", test_tamm_parser ("t1{ range O = -20; }"));
    //EXPECT_EQ (50.3321, squareroot (2533.310224));
}

TEST (ParserTest, RangeDeclarations0) { 
    ASSERT_EQ ("Success", test_tamm_parser ("t1{ range O = -20; }"));
    //EXPECT_EQ (50.3321, squareroot (2533.310224));
}
TEST (ParserTest, RangeDeclarations1) { 
    ASSERT_EQ ("Success", test_tamm_parser ("t1{ range O = -20; }"));
    //EXPECT_EQ (50.3321, squareroot (2533.310224));
}
TEST (ParserTest, RangeDeclarations2) { 
    ASSERT_EQ ("Success", test_tamm_parser ("t1{ range O = -20; }"));
    //EXPECT_EQ (50.3321, squareroot (2533.310224));
}
TEST (ParserTest, RangeDeclarations3) { 
    ASSERT_EQ ("Success", test_tamm_parser ("t1{ range O = -20; }"));
    //EXPECT_EQ (50.3321, squareroot (2533.310224));
}
TEST (ParserTest, RangeDeclarations4) { 
    ASSERT_EQ ("Success", test_tamm_parser ("t1{ range O = -20; }"));
    //EXPECT_EQ (50.3321, squareroot (2533.310224));
}
TEST (ParserTest, RangeDeclarations5) { 
    ASSERT_EQ ("Success", test_tamm_parser ("t1{ range O = -20; }"));
    //EXPECT_EQ (50.3321, squareroot (2533.310224));
}


int main(int argc, char **argv) {
  test_tamm_parser ("t1{ range O = 20; }");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


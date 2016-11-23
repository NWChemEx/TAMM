from __future__ import print_function


import os
import sys
from antlr4 import *
from collections import OrderedDict
from OpMinLexer import OpMinLexer
from OpMinParser import OpMinParser


stmt_refs = []
inputarrs = dict()
add_stmts = []
mult_stmts = []
indent = 0
tensor_decls = OrderedDict()
add_mult_order = OrderedDict()
destroy_temps = OrderedDict()
temps = OrderedDict()
func_offsets = []
array_decls = []

def printres(s):
    print(s, end="")

def printresws(s):
    print(" " + str(s), end=" ")

def printws():
    print(" ", end="")

def printnl(s):
    print(s, end="\n")

def printresi(s):
    print("".ljust(indent)+s, end="")

def printnli(s):
    print("".ljust(indent)+s, end="\n")


class Unfactorize(ParseTreeVisitor):

    def __init__(self, x='tce', y='t'):
        self.function_prefix = x
        self.label_prefix = y

    # Visit a parse tree produced by OpMinParser#translation_unit.
    def visitTranslation_unit(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#compound_element_list_opt.
    def visitCompound_element_list_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#compound_element_list.
    def visitCompound_element_list(self, ctx):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpMinParser#compound_element_list_prime.
    def visitCompound_element_list_prime(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#compound_element.
    def visitCompound_element(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#element_list_opt.
    def visitElement_list_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#element_list.
    def visitElement_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#element_list_prime.
    def visitElement_list_prime(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#element.
    def visitElement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#assignment_statement.
    def visitAssignment_statement(self, ctx):
        global stmt_refs,inputarrs,add_stmts,mult_stmts,tensor_decls,add_mult_order,destroy_temps,temps,func_offsets
        stmt_label = str(ctx.children[0].children[0])
        func_sig = "void " + self.function_prefix + "_" + stmt_label + "_("

        func_offset_sig = "void offset_" + self.function_prefix + "_" + stmt_label + "_("


        self.visitChildren(ctx.children[2])
        lhs_ref = list(stmt_refs)
        del stmt_refs[:]

        self.visitChildren(ctx.children[4])

        lhs_ref = lhs_ref[0]
        lhs_aname = str(lhs_ref.children[0])
        if lhs_aname not in tensor_decls.keys(): tensor_decls[lhs_aname] = lhs_aname

        if lhs_aname == stmt_label:
            temps[lhs_aname] = lhs_aname

            func_offset_sig +=  "Integer *l_" + lhs_aname + "_offset, Integer *k_" + lhs_aname + "_offset, Integer *size_" + lhs_aname+ ");"
            func_offsets.append(func_offset_sig)


        arefs = []
        #maxrhs = 0
        for c in stmt_refs:
            if isinstance(c,OpMinParser.Array_referenceContext):
                aname = str(c.children[0])
                #maxrhs = max(maxrhs, len(c.children[2].children))
                if aname not in tensor_decls.keys(): tensor_decls[aname] = aname
                arefs.append(aname)


        for k in temps.keys():
            if k in arefs:
                destroy_temps[stmt_label] = k  #Destroy after current stmt



        if len(arefs) > 1:
            mult_stmts.append(stmt_label)
            add_mult_order[stmt_label] = "mult"
        else:
            add_stmts.append(stmt_label)
            add_mult_order[stmt_label] = "add"


        arefs.append(lhs_aname)

        for ar in arefs:
            func_sig += "Integer *d_" + ar + ", Integer *k_" + ar + "_offset,"
            if not ar.startswith(self.label_prefix+"_") and ar not in inputarrs.keys(): inputarrs[ar] = ar;

        func_sig = func_sig[:-1] + ");"
        printnli(func_sig)


        del stmt_refs[:]



    # Visit a parse tree produced by OpMinParser#declaration.
    def visitDeclaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#id_list_opt.
    def visitId_list_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#id_list.
    def visitId_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#id_list_prime.
    def visitId_list_prime(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#num_list.
    def visitNum_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#num_list_prime.
    def visitNum_list_prime(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#identifier.
    def visitIdentifier(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#numerical_constant.
    def visitNumerical_constant(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#range_declaration.
    def visitRange_declaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#index_declaration.
    def visitIndex_declaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_declaration.
    def visitArray_declaration(self, ctx):
        global array_decls
        aname = str(ctx.children[1].children[0].children[0])
        array_decls.append(aname)
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_structure_list.
    def visitArray_structure_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_structure_list_prime.
    def visitArray_structure_list_prime(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_structure.
    def visitArray_structure(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#permut_symmetry_opt.
    def visitPermut_symmetry_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#symmetry_group_list.
    def visitSymmetry_group_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#symmetry_group_list_prime.
    def visitSymmetry_group_list_prime(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#symmetry_group.
    def visitSymmetry_group(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#expansion_declaration.
    def visitExpansion_declaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#volatile_declaration.
    def visitVolatile_declaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#iteration_declaration.
    def visitIteration_declaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#statement.
    def visitStatement(self, ctx):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpMinParser#assignment_operator.
    def visitAssignment_operator(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#unary_expression.
    def visitUnary_expression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#primary_expression.
    def visitPrimary_expression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_reference.
    def visitArray_reference(self, ctx):
        stmt_refs.append(ctx)

    # Visit a parse tree produced by OpMinParser#expression.
    def visitExpression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#plusORminus.
    def visitPlusORminus(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#additive_expression.
    def visitAdditive_expression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#additive_expression_prime.
    def visitAdditive_expression_prime(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#multiplicative_expression.
    def visitMultiplicative_expression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#multiplicative_expression_prime.
    def visitMultiplicative_expression_prime(self, ctx):
        return self.visitChildren(ctx)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_stream = FileStream(sys.argv[1])
    else:
        printnl("Please provide an input file!")
        sys.exit(1)

    lexer = OpMinLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = OpMinParser(token_stream)
    tree = parser.translation_unit()

    fname = os.path.basename(sys.argv[1])
    fname = fname.split(".")[0]
    fname = fname.split("_")

    if (len(fname) != 2):
        printnl("File name should be of the form ccsd_t1.eq")
        sys.exit(1)

    print("/*")
    with open(sys.argv[1],'r') as f:
        for line in f:
            print(" *  " + line.strip("\n"))
    print("*/\n\n")

    visitor = Unfactorize(fname[0],fname[1])
    visitor.visit(tree)

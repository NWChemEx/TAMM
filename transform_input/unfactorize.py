from __future__ import print_function


import os
import sys
from antlr4 import *
from collections import OrderedDict
from OpMinLexer import OpMinLexer
from OpMinParser import OpMinParser

indent = 0
orig_ops = []
get_lhs_aref = []
get_rhs_aref = []
get_alpha = 1.0


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

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


# tc[tc_ids] = alpha * ta[ta_ids]
class AddOp:
    tc = ''
    ta = ''
    alpha = 0.0
    tc_ids = []
    ta_ids = []

    def __init__(self, tcname, taname, alp, tc_id, ta_id):
        self.tc = tcname
        self.ta = taname
        self.alpha = alp
        self.tc_ids = tc_id
        self.ta_ids = ta_id

    def printOp(self):
        op = self.tc + str(self.tc_ids) + " += " + str(self.alpha) + " * " + self.ta + str(self.ta_ids)
        printnl(op)


# tc[tc_ids] += alpha * ta[ta_ids] * tb[tb_ids]
class MultOp:
    tc = ''
    tb = ''
    ta = ''
    alpha = 0.0
    tc_ids = []
    ta_ids = []
    ta_ids = []

    def __init__(self, tcname, taname, tbname, alp, tc_id, ta_id, tb_id):
        self.tc = tcname
        self.ta = taname
        self.tb = tbname
        self.alpha = alp
        self.tc_ids = tc_id
        self.tb_ids = tb_id
        self.ta_ids = ta_id

    def printOp(self):
        op = self.tc + str(self.tc_ids) + " += " + str(self.alpha) + " * "
        op += self.ta + str(self.ta_ids) + " * " + self.tb + str(self.tb_ids)
        printnl(op)


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
        global orig_ops, get_lhs_aref, get_rhs_aref,get_alpha
        lhs = ctx.children[0]
        assignOp = ctx.children[1]
        rhs = ctx.children[2]
        if str(assignOp).strip() == ":":
            lhs = ctx.children[2]
            assignOp = ctx.children[3]
            rhs = ctx.children[4]

        self.visitChildren(lhs)
        lhs_aref = get_lhs_aref

        self.visitChildren(rhs)
        rhs_aref = get_rhs_aref

        get_lhs_aref = []
        get_rhs_aref = []

        ac_rhs = []
        for i in rhs_aref:
            if i[0]!=lhs_aref[0]:
                ac_rhs.append(i)

        lra = len(ac_rhs)
        assert lra==1 or lra==2

        newop = ''
        if lra == 1: newop = AddOp(lhs_aref[0],ac_rhs[0][0],get_alpha,lhs_aref[1],ac_rhs[0][1])
        elif lra == 2: newop = MultOp(lhs_aref[0],ac_rhs[0][0],ac_rhs[1][0],get_alpha,lhs_aref[1],ac_rhs[0][1],ac_rhs[1][1])
        assert newop

        get_alpha = 1.0
        orig_ops.append(newop)

        # printres(lhs_aref[0] + str(lhs_aref[1]))
        #printresws("+=")
        # printres(get_alpha)
        # for i in ac_rhs:
        #     printresws("*")
        #     printres(i[0] + str(i[1]))
        # print("")




    # Visit a parse tree produced by OpMinParser#array_reference.
    def visitArray_reference(self, ctx):
        global get_lhs_aref, get_rhs_aref
        if isinstance(ctx,OpMinParser.Array_referenceContext):
            aname = str(ctx.children[0]) #print arrayname
            ilist = self.visitId_list(ctx.children[2])
            get_lhs_aref = [aname,ilist]
            get_rhs_aref.append([aname,ilist])


    # Visit a parse tree produced by OpMinParser#id_list.
    def visitId_list(self, ctx):
        idecl = []
        idecl.append(str(ctx.children[0].children[0]))
        var = ctx.children[1]
        while (var.children):
            idecl.append(str(var.children[1].children[0]))
            var = var.children[2]
        return idecl


    # Visit a parse tree produced by OpMinParser#identifier.
    def visitIdentifier(self, ctx):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpMinParser#declaration.
    def visitDeclaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#id_list_opt.
    def visitId_list_opt(self, ctx):
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

    # Visit a parse tree produced by OpMinParser#numerical_constant.
    def visitNumerical_constant(self, ctx):
        global get_alpha
        nc = str(ctx.children[0])
        get_alpha = get_alpha * convert_to_float(nc)
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#range_declaration.
    def visitRange_declaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#index_declaration.
    def visitIndex_declaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_declaration.
    def visitArray_declaration(self, ctx):
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
        global get_alpha
        if ctx.children:
            operator = str(ctx.children[0]).strip()
            if operator == "-":
                get_alpha = -1.0
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#primary_expression.
    def visitPrimary_expression(self, ctx):
        return self.visitChildren(ctx)


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

    for op in orig_ops:
        op.printOp()


    print("")

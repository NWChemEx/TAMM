from __future__ import print_function

import sys
import math
from antlr4 import *
from OpMinLexer import OpMinLexer
from OpMinParser import OpMinParser


single_use_temps = dict()

arefs = []
constants = -999
range_vals = dict()

class CollectTemps(ParseTreeVisitor):

    def __init__(self):
        self.t2tEq = ""
        self.ioInd = ""
        sys.setrecursionlimit(67108864)

    # Visit a parse tree produced by OpMinParser#translation_unit.
    def visitTranslation_unit(self, ctx):
        self.visitChildren(ctx)
        assert(len(self.ioInd) == 1)
        return [single_use_temps, "[" + self.ioInd[0]]



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


    # Visit a parse tree produced by OpMinParser#declaration.
    def visitDeclaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#id_list_opt.
    def visitId_list_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#id_list.
    def visitId_list(self, ctx):
        idecl = []
        idecl.append(str(ctx.children[0].children[0]))
        var = ctx.children[1]
        while (var.children):
            idecl.append(str(var.children[1].children[0]))
            var = var.children[2]
        return idecl


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

    # Visit a parse tree produced by OpMinParser#range_declaration.
    def visitRange_declaration(self, ctx):
        global range_vals
        rv = float(ctx.children[3].getText())
        rvars = ctx.children[1].getText().split(",")
        self.t2tEq += ("range ")
        self.t2tEq += (",".join(rvars))
        self.t2tEq += (" = ")
        rv = int(rv)
        self.t2tEq += str(rv)
        self.t2tEq += (";\n")
        for r in rvars:
            range_vals[r.strip()] = rv



    # Visit a parse tree produced by OpMinParser#index_declaration.
    def visitIndex_declaration(self, ctx):
        self.t2tEq += (ctx.children[0].getText())
        self.t2tEq += " " + (ctx.children[1].getText()) + " "
        self.t2tEq += ("= ")
        self.t2tEq += (ctx.children[3].getText())
        self.t2tEq += (";\n")


    # Visit a parse tree produced by OpMinParser#array_declaration.
    def visitArray_declaration(self, ctx):
        self.t2tEq += (ctx.children[0].getText())
        self.t2tEq += (" " + ctx.children[1].getText())

        if (len(ctx.children) > 3):
            self.t2tEq += (": " + ctx.children[3].getText())
        self.t2tEq += (";\n")


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



    # Visit a parse tree produced by OpMinParser#numerical_constant.
    def visitNumerical_constant(self, ctx):
        global constants
        nv = str(ctx.children[0])
        if "/" not in nv: nv = float(nv)
        else:
            num = nv.split("/")
            nv = float(num[0])*1.0/float(num[1])

        constants = constants * nv


    # Visit a parse tree produced by OpMinParser#assignment_statement.
    def visitAssignment_statement(self, ctx):
        global arefs
        op_label = ""
        lhs = ctx.children[0]
        aop = ctx.children[1]
        rhs = ctx.children[2]
        if isinstance(lhs,OpMinParser.IdentifierContext):
            op_label = lhs.getText()
            lhs = ctx.children[2]
            aop = ctx.children[3]
            rhs = ctx.children[4]

        lhs = lhs.getText()
        aop = aop.getText()

        arefs = []

        self.visitExpression(rhs)

        num_arr = len(arefs)
        assert(num_arr==1 or num_arr == 2)


        lhs_aname = lhs.split("[")[0]
        lhs_aname = lhs_aname.strip()
        if lhs_aname not in single_use_temps and "i0" not in lhs_aname:
            single_use_temps[lhs_aname] = []

        if lhs_aname == "i0":
            self.ioInd = lhs.split("[")[1:]
            for a in arefs:
                if a in single_use_temps:
                    rhstext = rhs.getText()
                    ra = 1.0
                    rc = rhstext.split("*")[0]
                    try:
                        ra = float(rc.strip())
                    except:
                        pass
                    single_use_temps[a].append(ra)


    # Visit a parse tree produced by OpMinParser#assignment_operator.
    def visitAssignment_operator(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#unary_expression.
    def visitUnary_expression(self, ctx):
        global constants
        if str(ctx.children[0]) == "-": constants = -1;
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#primary_expression.
    def visitPrimary_expression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_reference.
    def visitArray_reference(self, ctx):
        aname = str(ctx.children[0])
        arefs.append(aname)
        #arefInd.append(self.visitId_list_opt(ctx.children[2]))


    # Visit a parse tree produced by OpMinParser#expression.
    def visitExpression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#plusORminus.
    def visitPlusORminus(self, ctx):
        self.t2tEq += (ctx.children[0].getText())
        #return self.visitChildren(ctx)


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




class MinimizeTemps(ParseTreeVisitor):

    def __init__(self, x, ind):
        self.t2tEq = ""
        self.temps = x
        self.ioInd = ind
        sys.setrecursionlimit(67108864)

    # Visit a parse tree produced by OpMinParser#translation_unit.
    def visitTranslation_unit(self, ctx):
        self.visitChildren(ctx)
        return self.t2tEq
        # global  range_vals
        # print(range_vals)


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


    # Visit a parse tree produced by OpMinParser#declaration.
    def visitDeclaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#id_list_opt.
    def visitId_list_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#id_list.
    def visitId_list(self, ctx):
        idecl = []
        idecl.append(str(ctx.children[0].children[0]))
        var = ctx.children[1]
        while (var.children):
            idecl.append(str(var.children[1].children[0]))
            var = var.children[2]
        return idecl


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

    # Visit a parse tree produced by OpMinParser#range_declaration.
    def visitRange_declaration(self, ctx):
        global range_vals
        rv = float(ctx.children[3].getText())
        rvars = ctx.children[1].getText().split(",")
        self.t2tEq += ("range ")
        self.t2tEq += (",".join(rvars))
        self.t2tEq += (" = ")
        rv = int(rv)
        self.t2tEq += str(rv)
        self.t2tEq += (";\n")
        for r in rvars:
            range_vals[r.strip()] = rv



    # Visit a parse tree produced by OpMinParser#index_declaration.
    def visitIndex_declaration(self, ctx):
        self.t2tEq += (ctx.children[0].getText())
        self.t2tEq += " " + (ctx.children[1].getText()) + " "
        self.t2tEq += ("= ")
        self.t2tEq += (ctx.children[3].getText())
        self.t2tEq += (";\n")


    # Visit a parse tree produced by OpMinParser#array_declaration.
    def visitArray_declaration(self, ctx):
        self.t2tEq += (ctx.children[0].getText())
        self.t2tEq += (" " + ctx.children[1].getText())

        if (len(ctx.children) > 3):
            self.t2tEq += (": " + ctx.children[3].getText())
        self.t2tEq += (";\n")


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



    # Visit a parse tree produced by OpMinParser#numerical_constant.
    def visitNumerical_constant(self, ctx):
        global constants
        nv = str(ctx.children[0])
        if "/" not in nv: nv = float(nv)
        else:
            num = nv.split("/")
            nv = float(num[0])*1.0/float(num[1])

        constants = constants * nv


    # Visit a parse tree produced by OpMinParser#assignment_statement.
    def visitAssignment_statement(self, ctx):
        global arefs
        op_label = ""
        lhs = ctx.children[0]
        aop = ctx.children[1]
        rhs = ctx.children[2]
        if isinstance(lhs,OpMinParser.IdentifierContext):
            op_label = lhs.getText()
            lhs = ctx.children[2]
            aop = ctx.children[3]
            rhs = ctx.children[4]

        lhs = lhs.getText()
        aop = aop.getText()

        arefs = []

        self.visitExpression(rhs)

        num_arr = len(arefs)
        assert(num_arr==1 or num_arr == 2)

        lhs_aname = lhs.split("[")[0]
        lhs_aname = lhs_aname.strip()

        rhstext = rhs.getText()
        ra = 1.0
        rc = rhstext.split("*")[0]
        rexp = rhstext.split("*")[1:]
        try:
            ra = float(rc.strip())
        except:
            pass

        rhs_expression = ""
        for r_exp in rexp: rhs_expression += " * " + r_exp

        #if in temps, these should be replaced with i0 +=
        if lhs_aname in self.temps:
            astmt = ""
            if op_label: astmt += op_label + " : "
            astmt += "i0" + self.ioInd + " += "

            if self.temps[lhs_aname]: ra = ra * (self.temps[lhs_aname])[0]
            astmt += str(ra)
            astmt += rhs_expression

            self.t2tEq += (astmt) + ";\n"
        else:
            remove_stmt = False
            for arr in arefs:
                rhs_aname = arr.split("[")[0]
                rhs_aname = rhs_aname.strip()
                if rhs_aname in self.temps:
                    remove_stmt = True
            if not remove_stmt:
                astmt = ""
                if op_label: astmt += op_label + " : "
                astmt += lhs + " " + aop + " " + str(ra) + rhs_expression
                self.t2tEq += astmt + ";\n"



    # Visit a parse tree produced by OpMinParser#assignment_operator.
    def visitAssignment_operator(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#unary_expression.
    def visitUnary_expression(self, ctx):
        global constants
        if str(ctx.children[0]) == "-": constants = -1;
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#primary_expression.
    def visitPrimary_expression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_reference.
    def visitArray_reference(self, ctx):
        aname = str(ctx.children[0])
        arefs.append(aname)
        #arefInd.append(self.visitId_list_opt(ctx.children[2]))


    # Visit a parse tree produced by OpMinParser#expression.
    def visitExpression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#plusORminus.
    def visitPlusORminus(self, ctx):
        self.t2tEq += (ctx.children[0].getText())
        #return self.visitChildren(ctx)


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


#TAMM TO TAMM

arefs = []
arefInd = []
constants = -999
range_vals = dict()


def is_substr(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if find not in data[i]:
            return False
    return True


def long_substr(data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0]) - i + 1):
                if j > len(substr) and is_substr(data[0][i:i + j], data):
                    substr = data[0][i:i + j]
    return substr


class TAMMtoTAMM(ParseTreeVisitor):

    def __init__(self, x):
        self.mdoption = x
        self.t2tEq = ""
        sys.setrecursionlimit(67108864)

    # Visit a parse tree produced by OpMinParser#translation_unit.
    def visitTranslation_unit(self, ctx):
        self.visitChildren(ctx)
        return self.t2tEq
        # global  range_vals
        # print(range_vals)


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


    # Visit a parse tree produced by OpMinParser#declaration.
    def visitDeclaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#id_list_opt.
    def visitId_list_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#id_list.
    def visitId_list(self, ctx):
        idecl = []
        idecl.append(str(ctx.children[0].children[0]))
        var = ctx.children[1]
        while (var.children):
            idecl.append(str(var.children[1].children[0]))
            var = var.children[2]
        return idecl


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

    # Visit a parse tree produced by OpMinParser#range_declaration.
    def visitRange_declaration(self, ctx):
        global range_vals
        rv = float(ctx.children[3].getText())
        rvars = ctx.children[1].getText().split(",")
        self.t2tEq += ("range ")
        self.t2tEq += (",".join(rvars))
        self.t2tEq += (" = ")
        rv = int(rv)
        self.t2tEq += str(rv)
        self.t2tEq += (";\n")
        for r in rvars:
            range_vals[r.strip()] = rv



    # Visit a parse tree produced by OpMinParser#index_declaration.
    def visitIndex_declaration(self, ctx):
        self.t2tEq += (ctx.children[0].getText())
        self.t2tEq += " " + (ctx.children[1].getText()) + " "
        self.t2tEq += ("= ")
        self.t2tEq += (ctx.children[3].getText())
        self.t2tEq += (";\n")


    # Visit a parse tree produced by OpMinParser#array_declaration.
    def visitArray_declaration(self, ctx):
        self.t2tEq += (ctx.children[0].getText())
        self.t2tEq += (" " + ctx.children[1].getText())

        if (len(ctx.children) > 3):
            self.t2tEq += (": " + ctx.children[3].getText())
        self.t2tEq += (";\n")


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



    # Visit a parse tree produced by OpMinParser#numerical_constant.
    def visitNumerical_constant(self, ctx):
        global constants
        nv = str(ctx.children[0])
        if "/" not in nv: nv = float(nv)
        else:
            num = nv.split("/")
            nv = float(num[0])*1.0/float(num[1])

        constants = constants * nv


    # Visit a parse tree produced by OpMinParser#assignment_statement.
    def visitAssignment_statement(self, ctx):
        global arefs, arefInd, constants
        op_label = ""
        lhs = ctx.children[0]
        aop = ctx.children[1]
        rhs = ctx.children[2]
        if isinstance(lhs,OpMinParser.IdentifierContext):
            op_label = lhs.getText()
            lhs = ctx.children[2]
            aop = ctx.children[3]
            rhs = ctx.children[4]

        lhs = lhs.getText()
        aop = aop.getText()

        astmt = lhs +  " " + aop + " "
        if op_label: astmt = op_label + " : " + astmt

        arefs = []
        arefInd = []
        constants = 1

        self.visitExpression(rhs)

        self.t2tEq += (astmt)

        num_arr = len(arefs)

        assert(num_arr==1 or num_arr == 2)

        symm_fact = 1.0
        if num_arr == 2:
            t1i = arefInd[0]
            t2i = arefInd[1]
            t1iu = t1i[0:len(t1i)/2]
            t1il = t1i[len(t1i)/2:]

            t2iu = t2i[0:len(t2i)/2]
            t2il = t2i[len(t2i)/2:]

            t1hu = []
            t1hl = []
            t1pu = []
            t1pl = []
            t2hu = []
            t2hl = []
            t2pu = []
            t2pl = []

            for i in t1iu:
                if i[0] == 'h': t1hu.append(i)
                elif i[0] == 'p': t1pu.append(i)
            for i in t1il:
                if i[0] == 'h': t1hl.append(i)
                elif i[0] == 'p': t1pl.append(i)

            for i in t2iu:
                if i[0] == 'h': t2hu.append(i)
                elif i[0] == 'p': t2pu.append(i)
            for i in t2il:
                if i[0] == 'h': t2hl.append(i)
                elif i[0] == 'p': t2pl.append(i)

            t1hu = set(t1hu)
            t1hl = set(t1hl)
            t1pu = set(t1pu)
            t1pl = set(t1pl)
            t2hu = set(t2hu)
            t2hl = set(t2hl)
            t2pu = set(t2pu)
            t2pl = set(t2pl)

            symm_fact *= math.factorial(len(t1hu.intersection(t2hu)))
            symm_fact *= math.factorial(len(t1hu.intersection(t2hl)))
            symm_fact *= math.factorial(len(t1hl.intersection(t2hu)))
            symm_fact *= math.factorial(len(t1hl.intersection(t2hl)))

            symm_fact *= math.factorial(len(t1pu.intersection(t2pu)))
            symm_fact *= math.factorial(len(t1pu.intersection(t2pl)))
            symm_fact *= math.factorial(len(t1pl.intersection(t2pu)))
            symm_fact *= math.factorial(len(t1pl.intersection(t2pl)))



            if self.mdoption == 0: constants = constants*symm_fact
            elif self.mdoption > 0: constants = float(constants*1.0)/(symm_fact)

        self.t2tEq += (str(constants))
        for ar in range(0,num_arr):
            self.t2tEq += (" * " + arefs[ar] + "[" + str(",".join(arefInd[ar])) + "]")
        self.t2tEq += (";\n")


    # Visit a parse tree produced by OpMinParser#assignment_operator.
    def visitAssignment_operator(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#unary_expression.
    def visitUnary_expression(self, ctx):
        global constants
        if str(ctx.children[0]) == "-": constants = -1;
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#primary_expression.
    def visitPrimary_expression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_reference.
    def visitArray_reference(self, ctx):
        aname = str(ctx.children[0])
        arefs.append(aname)
        arefInd.append(self.visitId_list_opt(ctx.children[2]))


    # Visit a parse tree produced by OpMinParser#expression.
    def visitExpression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#plusORminus.
    def visitPlusORminus(self, ctx):
        self.t2tEq += (ctx.children[0].getText())
        #return self.visitChildren(ctx)


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


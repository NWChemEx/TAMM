from __future__ import print_function

import sys
from antlr4 import *
from OpMinLexer import OpMinLexer
from OpMinParser import OpMinParser
from antlr4.tree.Tree import TerminalNodeImpl
from collections import OrderedDict

#self.function_prefix = "icsd"
#self.label_prefix = 't2'
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


class OpMinVisitor(ParseTreeVisitor):

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
        global inputarrs,add_stmts,mult_stmts,indent,tensor_decls,add_mult_order,func_offsets,array_decls
        printnl("extern \"C\" {")
        indent += 2
        self.visitChildren(ctx)
        printnl("")
        for fo in func_offsets:
            printnli(fo)
        indent -= 2
        printnl("}")

        printnl("\nnamespace ctce {\n")

        declare_lib_api = "void schedule_linear(std::vector<Tensor> &tensors, std::vector<Operation> &ops);\n"
        declare_lib_api += "".ljust(indent)+"void schedule_linear_lazy(std::vector<Tensor> &tensors, std::vector<Operation> &ops);\n"
        declare_lib_api += "".ljust(indent)+"void schedule_levels(std::vector<Tensor> &tensors, std::vector<Operation> &ops);\n"

        printnli(declare_lib_api)
        printnli("extern \"C\" {")
        indent += 2

        printresi("void " + self.function_prefix + "_" + self.label_prefix + "_cxx_(")
        func_sig = ""
        for ia in inputarrs:
            func_sig += "Integer *d_" + ia + ", "

        func_sig += "\n" + "".ljust(indent)

        for ia in inputarrs:
            func_sig += "Integer *k_" + ia + "_offset, "

        func_sig = func_sig[:-2]
        printres(func_sig)
        printnl(") {\n")


        printnli("static bool set_" + self.label_prefix + " = true;")
        printnli("")

        for ast in add_stmts:
            printnli("Assignment op_"+ ast + ";")

        for mst in mult_stmts:
            printnli("Multiplication op_" + mst + ";")

        printnli("")
        printnli("DistType idist = (Variables::intorb()) ? dist_nwi : dist_nw;")
        printnli("static Equations eqs;\n")
        printnli("if (set_"+ self.label_prefix + ") {")
        printnli("  " + self.function_prefix + "_" + self.label_prefix + "_equations(eqs);")
        printnli("  set_"+ self.label_prefix + " = false;")
        printnli("}\n")

        printnli("std::vector <Tensor> tensors;")
        printnli("std::vector <Operation> ops;")

        printnli("tensors_and_ops(eqs, tensors, ops);")

        printnl("")
        ti = 0
        #for td in tensor_decls.keys():
        for td in array_decls:
            printnli("Tensor *" + td + " = &tensors[" + str(ti) + "];")
            ti += 1

        printnl("")

        printnli("/* ----- Insert attach code ------ */")
        printnli("v->set_dist(idist);")
        printnli("i0->attach(*k_i0_offset, 0, *d_i0);")
        printnli("f->attach(*k_f_offset, 0, *d_f);")
        printnli("v->attach(*k_v_offset, 0, *d_v);\n")

        printnli("#if 1")
        printnli("  schedule_levels(tensors, ops);")
        printnli("#else")

        indent += 2

        ti = 0
        for amo in add_mult_order:
            printnli("op_" +  amo + " = ops[" + str(ti) + "]." + add_mult_order[amo] + ";")
            ti += 1

        printnl("")

        for amo in add_mult_order:
            if amo in array_decls:
                printnli("CorFortran(1, " + amo + ", offset_" + self.function_prefix + "_" + amo + "_);")
            printnli("CorFortran(1, op_" +  amo + ", " + self.function_prefix + "_" + amo + "_);")

            if amo in destroy_temps.keys():
                printnli("destroy(" + destroy_temps[amo] + ");")

        indent -= 2
        printnli("#endif\n")

        printnli("/* ----- Insert detach code ------ */")
        printnli("f->detach();")
        printnli("i0->detach();")
        printnli("v->detach();")

        printnli("}")
        printnl("} // extern C")
        printnl("}; // namespace ctce")


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
        maxrhs = 0
        for c in stmt_refs:
            if isinstance(c,OpMinParser.Array_referenceContext):
                aname = str(c.children[0])
                maxrhs = max(maxrhs, len(c.children[2].children))
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



f_v_decls = dict()

class OpminOutToCTCE(ParseTreeVisitor):

    # Visit a parse tree produced by OpMinParser#translation_unit.
    def visitTranslation_unit(self, ctx):
        #printnl("{")
        self.visitChildren(ctx)
        #printnl("}")


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

        idecl = str(ctx.children[0].children[0])
        var = ctx.children[1]
        while (var.children):
            idecl += "," + str(var.children[1].children[0])
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


    # Visit a parse tree produced by OpMinParser#numerical_constant.
    def visitNumerical_constant(self, ctx):
        printresws(ctx.children[0])
        #return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#range_declaration.
    def visitRange_declaration(self, ctx):
        var = ctx.children[1]
        idecl = "range "
        idecl += self.visitId_list(var)
        value =  ctx.children[3].children[0]
        printnli(idecl + " = " + str(value) +";")

        #return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#index_declaration.
    def visitIndex_declaration(self, ctx):
        var = ctx.children[1]
        idecl = "index "
        idecl += self.visitId_list(var)
        value = ctx.children[3].children[0]
        printnli(idecl + " = " + str(value) + ";")
        self.visitChildren(ctx)


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
        global f_v_decls
        aname = str(ctx.children[0])
        renrange = False

        if aname.startswith("f_"):
            aname = 'f'
            renrange = True
        elif aname.startswith("v_"):
            aname = 'v'
            renrange = True

        if aname in f_v_decls: return
        else: f_v_decls[aname] = aname
        adecl = "array " + aname

        adecl += "["
        astruct = ""
        for c in ctx.children:
            if isinstance(c,OpMinParser.AstructContext):
                astruct = c



        upper = ""

        if astruct.children[1].children:
            upper = astruct.children[1].children[0]
            upper = self.visitId_list(upper)
            if renrange:
                nu = len(upper.split(","))
                adecl += "N,"*nu
                adecl = adecl[:-1]
            else: adecl += upper

        adecl += ']['
        lower = ""

        if astruct.children[4].children:
            lower = astruct.children[4].children[0]
            lower = self.visitId_list(lower)
            if renrange:
                nu = len(lower.split(","))
                adecl += "N,"*nu
                adecl = adecl[:-1]
            else: adecl += lower

        adecl += '];'
        printnli(adecl)


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


    # Visit a parse tree produced by OpMinParser#assignment_statement.
    def visitAssignment_statement(self, ctx):
        lhs = self.visitExpression(ctx.children[0])
        printres(ctx.children[1].children[0])
        rhs = self.visitExpression(ctx.children[2])
        printnl(";")



    # Visit a parse tree produced by OpMinParser#assignment_operator.
    def visitAssignment_operator(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#unary_expression.
    def visitUnary_expression(self, ctx):
        if str(ctx.children[0]) == "+": printres(" +")
        elif str(ctx.children[0]) == "-": printres(" -")
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#primary_expression.
    def visitPrimary_expression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_reference.
    def visitArray_reference(self, ctx):
        aname = str(ctx.children[0])
        if aname.startswith("f_"): aname = 'f';
        elif aname.startswith("v_"): aname = 'v';
        aref = (aname + "[")
        ilist = self.visitId_list(ctx.children[2])
        aref += (ilist) + "]"
        printresws(aref)

    # Visit a parse tree produced by OpMinParser#expression.
    def visitExpression(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#plusORminus.
    def visitPlusORminus(self, ctx):
        printres(ctx.children[0])
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
        if ctx.children:
            if str(ctx.children[0]) == "*": printres("*")
        return self.visitChildren(ctx)


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
lhs_ops = []
rhs_ops = []
range_decls = []
collect_array_decls = OrderedDict()
collect_index_decls = []

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


def printIndexList(il):
    if not il: return  ""
    ilstr = "["
    for i in il:
        ilstr += i
        if il[-1] != i: ilstr += ","
    ilstr += "]"
    return ilstr


def convertFV(tensor, ids):
    global collect_array_decls
    if tensor != "f" and tensor != "v": return tensor
    adims = []
    type = tensor + "_"
    for index in ids:
        if (index[0] == 'h'):
            type += 'o'
            adims.append("O")
        elif (index[0] == 'p'):
            type += 'v'
            adims.append("V")
        else:
            printres("arr index can only start with either p or h\n")

    dimLen = len(adims)
    if type not in collect_array_decls:
        adecl = "array " + type + "(["
        for l in adims[0:dimLen/2]:
            adecl += l + ","
        adecl = adecl[:-1]
        adecl += "]["
        for u in adims[dimLen/2:dimLen]:
            adecl += u + ","
        adecl = adecl[:-1]
        adecl += "]);"
        collect_array_decls[type] = adecl
    return type


class ExpandedOp:
    alpha = 1.0
    tensors = []
    tensor_ids = []

    def __init__(self):
        self.alpha = 1.0
        self.tensors = []
        self.tensor_ids = []

    def addFV(self):
        for i,t in enumerate(self.tensors):
            convertFV(t, self.tensor_ids[i])

    def printOp(self):
        op = ''
        if self.alpha > 0: op += "+"
        op += str(self.alpha)
        for i,t in enumerate(self.tensors):
            op += " * " + convertFV(t, self.tensor_ids[i]) + printIndexList(self.tensor_ids[i])
        printnl(op)


# tc[tc_ids] = alpha * ta[ta_ids]
class AddOp:
    tc = ''
    ta = ''
    alpha = 0.0
    tc_ids = []
    ta_ids = []
    expandedOp = ''

    def __init__(self, tcname, taname, alp, tc_id, ta_id):
        self.tc = tcname
        self.ta = taname
        self.alpha = alp
        self.tc_ids = tc_id
        self.ta_ids = ta_id

    def printOp(self):
        op = self.tc + printIndexList(self.tc_ids) + " += " + str(self.alpha) + " * " + self.ta + printIndexList(self.ta_ids)
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
    expandedOp = ''

    def __init__(self, tcname, taname, tbname, alp, tc_id, ta_id, tb_id):
        self.tc = tcname
        self.ta = taname
        self.tb = tbname
        self.alpha = alp
        self.tc_ids = tc_id
        self.tb_ids = tb_id
        self.ta_ids = ta_id

    def printOp(self):
        op = self.tc + printIndexList(self.tc_ids) + " += " + str(self.alpha) + " * "
        op += self.ta + printIndexList(self.ta_ids) + " * " + self.tb + printIndexList(self.tb_ids)
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
        global orig_ops, get_lhs_aref, get_rhs_aref,get_alpha,lhs_ops,rhs_ops
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

        if lhs_aref[0] not in lhs_ops: lhs_ops.append(lhs_aref[0])
        if ac_rhs[0][0] not in rhs_ops: rhs_ops.append(ac_rhs[0][0])
        if lra == 2:
            if ac_rhs[1][0] not in rhs_ops: rhs_ops.append(ac_rhs[1][0])

        get_alpha = 1.0
        orig_ops.append(newop)


    # Visit a parse tree produced by OpMinParser#array_reference.
    def visitArray_reference(self, ctx):
        global get_lhs_aref, get_rhs_aref
        if isinstance(ctx,OpMinParser.Array_referenceContext):
            aname = str(ctx.children[0]) #print arrayname
            ilist = self.visitId_list_opt(ctx.children[2])
            get_lhs_aref = [aname,ilist]
            get_rhs_aref.append([aname,ilist])


    # Visit a parse tree produced by OpMinParser#id_list.
    def visitId_list(self, ctx):
        idecl = []
#        if not ctx.children: return idecl
        # if len(ctx.children) == 1: return idecl
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
        global collect_index_decls, range_decls
        idecl = "index "
        ids =  (self.visitId_list(ctx.children[1]))
        for i in ids:
            idecl += i
            if ids[-1] != i: idecl += ", "

        rd = str(ctx.children[3].children[0])
        idecl += " = " + rd + ";"
        if rd not in range_decls: range_decls.append(rd)
        collect_index_decls.append(idecl)
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_declaration.
    def visitArray_declaration(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by OpMinParser#array_structure_list.
    def visitArray_structure_list(self, ctx):
        global collect_array_decls
        arrs = ctx.children[0]
        array_name = str(arrs.children[0])
        if array_name == "f" or array_name == "v": return self.visitChildren(ctx)

        array_struct = arrs.children[1]
        lower = self.visitChildren(array_struct.children[1])
        upper = self.visitChildren(array_struct.children[4])

        adecl = "array " + array_name + "(["
        if lower:
            for l in lower:
                adecl += l + ","
            adecl = adecl[:-1]
        adecl += "]["

        if upper:
            for u in upper:
                adecl += u + ","
            adecl = adecl[:-1]
        adecl += "]);"

        collect_array_decls[array_name] = adecl

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




def deleteOp(op,outputs):
    newout = []
    for o in outputs:
        if op.tc != o.tc:
            newout.append(o)

    return newout


def unfact(op,outputs):
    global orig_ops
    ta = op.ta

    unfacExp_a = ta
    unfacExp_b = ''
    isMultOp = False
    if isinstance(op,MultOp): isMultOp = True

    if isMultOp:
        tb = op.tb
        unfacExp_b = tb

    expandExp = []
    if unfacExp_a in lhs_ops or unfacExp_b in lhs_ops:
        for o in orig_ops:
            if o.tc == unfacExp_a or o.tc == unfacExp_b:
                expandExp.append(o)
                outputs = deleteOp(o,outputs)
        op.expandedOp = expandExp


    # if expandExp:
    #     for e in expandExp:
    #         e.printOp()
    # else: print("")


def expandOp(op):
    if not op.expandedOp:
        ep = ExpandedOp()
        ep.alpha = op.alpha
        ep.tensors.append(op.ta)
        ep.tensor_ids.append(op.ta_ids)
        if isinstance(op,MultOp):
            ep.tensors.append(op.tb)
            ep.tensor_ids.append(op.tb_ids)

        return [ep]

    expanded_ops = []
    for eop in op.expandedOp:
        eops = (expandOp(eop))
        expanded_ops.extend(eops)

    eop_all = []
    for eop in expanded_ops:
        ep = ExpandedOp()
        ep.alpha = op.alpha * eop.alpha

        ep.tensors.append(op.ta)
        ep.tensor_ids.append(op.ta_ids)

        for i, t in enumerate(eop.tensors):
            ep.tensors.append(t)
            ep.tensor_ids.append(eop.tensor_ids[i])

        eop_all.append(ep)

    return eop_all



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
    fname = fname.rsplit("_",1)

    methodName = fname[0]
    oplabel = fname[0]

    if len(fname) == 2:
        oplabel = fname[1]


    ci = 0
    for c in oplabel:
        if c.isdigit(): ci+=1
        else: break

    oplabel = oplabel[ci:]

    visitor = Unfactorize(methodName,oplabel)
    visitor.visit(tree)

    # for op in orig_ops:
    #     op.printOp()

    outputs = []
    for op in orig_ops:
        unfact(op,outputs)
        if op.tc in lhs_ops and op.tc not in rhs_ops:
            outputs.append(op)


    final_ops = OrderedDict()

    print("{\n")
    for decl in range_decls:
        print("range " + decl + " = 10;")
    #print("range N = 10;")
    print("")
    for decl in collect_index_decls:
        print(decl)

    print("")
    for o in outputs:
        final_ops[o] = (expandOp(o))

    for o in final_ops:
        for op in final_ops[o]:
            op.addFV()

    for decl in collect_array_decls:
        print(collect_array_decls[decl])

    print("")

    lhsOp = outputs[-1]
    printres(lhsOp.tc + printIndexList(lhsOp.tc_ids) + " = ")
    for o in final_ops:
        for op in final_ops[o]:
            op.printOp()

    print(";")
    print("\n}")


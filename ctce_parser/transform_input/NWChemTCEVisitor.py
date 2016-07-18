from __future__ import print_function

import sys
from antlr4 import *
from NWChemTCELexer import NWChemTCELexer
from NWChemTCEParser import NWChemTCEParser
from antlr4.tree.Tree import TerminalNodeImpl


def printres(s):
    print(s, end="")

def printresws(s):
    print(" " + str(s), end=" ")

def printws():
    print(" ", end="")


# This class defines a complete generic visitor for a parse tree produced by NWChemTCEParser.


labelcount_io = 0
labelcount_ia = dict()
namemap = dict()
array_decls = []
label_suffix = 't1' # label suffix comes from file name ?


class NWChemTCEVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by NWChemTCEParser#assignment_operator.
    def visitAssignment_operator(self, ctx):
        for c in ctx.children:
            printres(str(c))


    # Visit a parse tree produced by NWChemTCEParser#numerical_constant.
    def visitNumerical_constant(self, ctx):
        for c in ctx.children: printres(c)


    # Visit a parse tree produced by NWChemTCEParser#plusORminus.
    def visitPlusORminus(self, ctx):
        printres(str(ctx.children[0]))


    # Visit a parse tree produced by NWChemTCEParser#translation_unit.
    def visitTranslation_unit(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#compound_element_list_opt.
    def visitCompound_element_list_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#compound_element_list.
    def visitCompound_element_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#statement.
    def visitStatement(self, ctx):
        global labelcount_io, labelcount_ia
        #self.visitArray_reference(ctx.children[0])
        it = self.print_index_list(((ctx.children[0]).children[2]))
        ilist = it[1]
        lhs_array_name = str((ctx.children[0]).children[0])
        #printresws(lhs_array_name)
        label = ""

        if (lhs_array_name[0] != 'i'):
            printres("ARRAY NAME HAS TO START WITH an I \n")
            sys.exit(1)

        ino = int(lhs_array_name[1:])
        indentl = len(label_suffix+"_1")+8
        if (lhs_array_name == 'i0'):
            labelcount_io = labelcount_io + 1
            label = label_suffix + "_" + str(labelcount_io)
            for lname in labelcount_ia.keys():
                labelcount_ia[lname] = 0

        else:
            if lhs_array_name not in labelcount_ia.keys():
                labelcount_ia[lhs_array_name] = 1
            else: labelcount_ia[lhs_array_name] = labelcount_ia[lhs_array_name] + 1

            label = label_suffix
            for i in range(1,ino+1):
                label += "_" + str(labelcount_io+1)

            label +=  "_" + str(labelcount_ia[lhs_array_name])
            #labelcount_ia[lhs_array_name] = labelcount_ia[lhs_array_name] + 1

        printres(label + ":".ljust(indentl-len(label)) + lhs_array_name)  # Fix: use label instead of lhs_array_name
        printres("[")
        printres(ilist)
        printres("]")

        # print rhs now
        printws()
        self.visitAssignment_operator(ctx.children[1])
        printws()
        self.visitPtype(ctx.children[2])
        printres("\n")
        #self.visitChildren(ctx)


        # Visit a parse tree produced by NWChemTCEParser#perm.
    def visitPerm(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#ptype.
    def visitPtype(self, ctx):
        pchildren = []
        i =  0
        for c in (ctx.children):
            if isinstance(c,NWChemTCEParser.PermContext) or isinstance(c,NWChemTCEParser.SumExpContext):
                del pchildren[i-1]
                i = i-1
            else:
                pchildren.append(c)
                i = i+1

        for c in pchildren:
            if str(c) == '*': printresws(c)
            else: self.visit(c)


        printres(";")


    # Visit a parse tree produced by NWChemTCEParser#sumExp.
    def visitSumExp(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#id_list.
    def visitId_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#identifier.
    def visitIdentifier(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#array_reference.
    def visitArray_reference(self, ctx):
        if isinstance(ctx,NWChemTCEParser.Array_referenceContext):
            aname = str(ctx.children[0]) #print arrayname
            it = self.print_index_list((ctx.children[2])) #print indices
            atype = it[0]
            ilist = it[1]
            if (aname[0] == 'i' or aname[0] == 'f' or aname[0] == 'v'): printres(aname)
            else: printres(aname + "_" + atype)
            printres("[")
            printres(ilist)
            printres("]")
        #return self.visitChildren(ctx)



    def print_index_list(self, idxlist):
        il = ""
        type = ""
        for idx in idxlist.children:
            index = str(idx.children[0])
            il += index + ","
            if (index[0] == 'h'): type+= 'o'
            elif (index[0] == 'p'): type += 'v'
            else: printres("arr index can only start with either p or h\n")


        return [type,(il[:-1])]



stmt_seen = []
stmt_postpone = dict()
exec_order = []
uniqArrDecls = dict()
hind = 0
pind = 0

class NWChemTCEVisitorExecOrder(ParseTreeVisitor):

    # Visit a parse tree produced by NWChemTCEParser#assignment_operator.
    def visitAssignment_operator(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#numerical_constant.
    def visitNumerical_constant(self, ctx):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NWChemTCEParser#plusORminus.
    def visitPlusORminus(self, ctx):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by NWChemTCEParser#translation_unit.
    def visitTranslation_unit(self, ctx):
        self.visitChildren(ctx)
        return [exec_order,array_decls,[hind,pind]]


    # Visit a parse tree produced by NWChemTCEParser#compound_element_list_opt.
    def visitCompound_element_list_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#compound_element_list.
    def visitCompound_element_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#statement.
    def visitStatement(self, ctx):
        global stmt_seen, exec_order
        lhs_array_name = str((ctx.children[0]).children[0])
        stmt_seen.append(lhs_array_name)
        if (lhs_array_name in stmt_seen and lhs_array_name in stmt_postpone.keys()):
            addExec = stmt_postpone[lhs_array_name]
            exec_order.append(addExec)
            del stmt_postpone[lhs_array_name]
        rhsrefs = self.visitPtype(ctx.children[2])
        if not rhsrefs:
            exec_order.append(ctx)
            return
        stmt_postpone[lhs_array_name] = ctx

        # Visit a parse tree produced by NWChemTCEParser#perm.
    def visitPerm(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#ptype.
    def visitPtype(self, ctx):
        pchildren = []
        i =  0
        rhsrefs = []
        for c in (ctx.children):
            if isinstance(c,NWChemTCEParser.Array_referenceContext):
                self.visitArray_reference(c)
                aname = str(c.children[0])
                if aname[0] == 'i':
                    rhsrefs.append(aname)

        return rhsrefs


    # Visit a parse tree produced by NWChemTCEParser#sumExp.
    def visitSumExp(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#id_list.
    def visitId_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#identifier.
    def visitIdentifier(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NWChemTCEParser#array_reference.
    def visitArray_reference(self, ctx):
        global array_decls
        if isinstance(ctx,NWChemTCEParser.Array_referenceContext):
            aname = str(ctx.children[0]) #print arrayname
            if (aname[0] != 'i'):
                it = self.get_array_type((ctx.children[2])) #print indices

                atype = (it)
                if aname[0]=='f' or aname[0] == 'v': atype = 'N'*len(it)

                al = len(atype)
                upper = atype[0:al/2]
                lower = atype[al/2:al]
                upper = ",".join(upper)
                lower = ",".join(lower)

                renameArr = aname + "_" + (atype).lower()
                if aname[0] == 'f' or aname[0] == 'v': renameArr = aname
                decl = "array " + renameArr + "[" + upper + "]" + "[" + lower + "]: irrep" + str((ctx.children[4]).children[0]) + ";"
                if not renameArr in uniqArrDecls.keys():
                    uniqArrDecls[renameArr] = renameArr
                    array_decls.append(decl)

        #return self.visitChildren(ctx)


    def get_array_type(self, idxlist):
        global hind, pind
        type = ""
        for idx in idxlist.children:
            index = str(idx.children[0])
            if (index[0] == 'h'):
                type += 'O'
                hind = max(hind,int(index[1:]))
            elif (index[0] == 'p'):
                type += 'V'
                pind = max(pind, int(index[1:]))
            else:
                printres("arr index can only start with either p or h\n")

        return type
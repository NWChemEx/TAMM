from __future__ import print_function

import sys
from antlr4 import *
from SoTCELexer import SoTCELexer
from SoTCEParser import SoTCEParser
from antlr4.tree.Tree import TerminalNodeImpl


iexp = ""

def printres(s):
    global iexp
    iexp += str(s)

def printresws(s):
    global iexp
    iexp += " " + str(s) + " "



# This class defines a complete generic visitor for a parse tree produced by SoTCEParser.

uniqArrDecls = dict()
array_decls = []
pind = 0
hind = 0

class SoTCEVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by SoTCEParser#numerical_constant.
    def visitNumerical_constant(self, ctx):
        printres(float(str(ctx.children[0])))


    # Visit a parse tree produced by SoTCEParser#plusORminus.
    def visitPlusORminus(self, ctx):
        if ctx.op.type == SoTCELexer.PLUS:
            printres(" + ")
        else:  printres(" - ")


    # Visit a parse tree produced by SoTCEParser#translation_unit.
    def visitTranslation_unit(self, ctx):
        global array_decls,iexp
        self.visitChildren(ctx)
        return [array_decls,iexp]


    # Visit a parse tree produced by SoTCEParser#compound_element_list_opt.
    def visitCompound_element_list_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SoTCEParser#compound_element_list.
    def visitCompound_element_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SoTCEParser#factors.
    def visitFactors(self, ctx):
        printres("\n")
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SoTCEParser#factors_opt.
    def visitFactors_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SoTCEParser#ptype.
    def visitPtype(self, ctx):
        global uniqArrDecls
        if isinstance(ctx.children[0],SoTCEParser.PlusORminusContext): self.visitChildren(ctx)
        elif str(ctx.children[0]) == "*":
            if not isinstance(ctx.children[1], SoTCEParser.SumExpContext):
                printresws("*")
                aname = (str(ctx.children[1]))

                it = self.visitArrDims(ctx.children[2])

                adims = it[0]
                atype = it[1]

                al = len(atype)
                if aname[0] == 'f' or aname[0] == 'v': atype = 'N' * al


                upper = atype[0:al/2]
                lower = atype[al/2:al]
                upper = ",".join(upper)
                lower = ",".join(lower)

                renameArr = aname + "_" + (atype).lower()
                if aname[0] == 'f' or aname[0] == 'v': renameArr = aname

                printres(renameArr)
                printres(adims)
                decl = "array " + renameArr + "[" + upper + "]" + "[" + lower + "];"
                if not renameArr in uniqArrDecls.keys():
                    uniqArrDecls[renameArr] = renameArr
                    array_decls.append(decl)



    # Visit a parse tree produced by SoTCEParser#arrDims.
    def visitArrDims(self, ctx):
        global  pind, hind
        adims = ""
        for c in ctx.children:
            s = str(c)

            if s == "(": adims += "["
            elif s == ")":
                adims = adims[:-1]
                adims += "]"
            else: adims += s + ","



        type = ""
        for c in ctx.children:
            index = str(c)
            if (index[0] == 'h'):
                type += 'O'
                hind = max(hind,int(index[1:]))
            elif (index[0] == 'p'):
                type += 'V'
                pind = max(pind, int(index[1:]))


        return [adims,type]


    # Visit a parse tree produced by SoTCEParser#sumExp.
    def visitSumExp(self, ctx):
        return self.visitChildren(ctx)



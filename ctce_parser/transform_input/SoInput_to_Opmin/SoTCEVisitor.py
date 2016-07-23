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
first_aref = ""
second_aref = ""
deleteInd = dict()
far = 0
permute_flag = []

class SoTCEVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by SoTCEParser#numerical_constant.
    def visitNumerical_constant(self, ctx):
        self.visitChildren(ctx)


    # Visit a parse tree produced by SoTCEParser#plusORminus.
    def visitPlusORminus(self, ctx):
        self.visitChildren(ctx)


    # Visit a parse tree produced by SoTCEParser#translation_unit.
    def visitTranslation_unit(self, ctx):
        global array_decls,iexp,first_aref,second_aref
        self.visitChildren(ctx)
        first_aref = first_aref[1:-1].split(",")

        newfirstref = first_aref
        
        if second_aref:
            second_aref = second_aref[1:-1].split(",")

            fsrefs= [first_aref,second_aref]
            newfirstref = set().union(*fsrefs)
            inter = set(first_aref).intersection(second_aref)
            newfirstref = set(newfirstref).difference(inter)

        return [array_decls,iexp,newfirstref,[pind,hind]]


    # Visit a parse tree produced by SoTCEParser#compound_element_list_opt.
    def visitCompound_element_list_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SoTCEParser#compound_element_list.
    def visitCompound_element_list(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SoTCEParser#factors.
    def visitFactors(self, ctx):
        global far;
        printres("\n")
        far += 1
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SoTCEParser#factors_opt.
    def visitFactors_opt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SoTCEParser#ptype.
    def visitPtype(self, ctx):
        global uniqArrDecls, permute_flag
        if isinstance(ctx.children[0],SoTCEParser.PlusORminusContext):
            if str(ctx.children[0].children[0]) == "+":
                printres(" + ")
            else:
                printres(" - ")
            if isinstance(ctx.children[1], SoTCEParser.Numerical_constantContext):
                permute_flag.append(str(ctx.children[1].children[0]))

        elif str(ctx.children[0]) == "*":
            if not isinstance(ctx.children[1], SoTCEParser.SumExpContext):
                aname = (str(ctx.children[1]))

                if aname == "P":
                    #permute_flag = permute_flag[:-1]
                    for n in permute_flag:
                        printres(n)
                    permute_flag = []
                    return

                for n in permute_flag:
                    printres(n)
                permute_flag = []
                printresws("*")


                it = self.visitArrDims(ctx.children[2])

                adims = it[0]
                atype = it[1]

                #al = len(atype)

                atypel = atype.split(",")
                al = len(atypel)
                #if aname[0] == 'f' or aname[0] == 'v': atype = 'N' * al

                upper = atypel[0:al/2]
                lower = atypel[al/2:al]

                newup = []
                newlow = []
                for u in range(0,len(upper)):
                    if upper[u][-1] != "*": newup.append(upper[u])

                for u in range(0,len(lower)):
                    if lower[u][-1] != "*": newlow.append(lower[u])

                upper = ",".join(newup)
                lower = ",".join(newlow)

                newat = []
                for at in range(0,len(atypel)):
                    if atypel[at][-1] != "*": newat.append(atypel[at])



                renameArr = aname + "_" + ("".join(newat)).lower()
                #if aname[0] == 'f' or aname[0] == 'v': renameArr = aname

                printres(renameArr)
                printres(adims)


                decl = "array " + renameArr + "([" + upper + "]" + "[" + lower + "]);"
                #if not first_aref: first_aref = adims
                #elif not second_aref and far == 1: second_aref = adims

                if not renameArr in uniqArrDecls.keys():
                    uniqArrDecls[renameArr] = renameArr
                    array_decls.append(decl)



    # Visit a parse tree produced by SoTCEParser#arrDims.
    def visitArrDims(self, ctx):
        global  pind, hind, first_aref, second_aref, far
        adims = ""
        ldims = "["
        for c in ctx.children:
            s = str(c)

            if s == "(":
                adims += "["
            elif s == ")":
                adims = adims[:-1]
                adims += "]"
            else:
                if s[-1] != "*":
                    adims += s + ","
                ldims += s + ","


        ldims += ']'
        if not first_aref:
            first_aref = ldims[:-1]
        elif not second_aref and far == 1:
            second_aref = ldims[:-1]

        type = ""
        for c in ctx.children:
            index = str(c)
            if (index[0] == 'h'):

                if index[-1] =="*":
                    hind = max(hind,int(index[1:-1]))
                    type += 'O*,'
                else:
                    hind = max(hind,int(index[1:]))
                    type += 'O,'

            elif (index[0] == 'p'):

                if index[-1] == "*":
                    pind = max(pind, int(index[1:-1]))
                    type += 'V*,'
                else:
                    pind = max(pind, int(index[1:]))
                    type += 'V,'

        return [adims,type[:-1]]


    # Visit a parse tree produced by SoTCEParser#sumExp.
    def visitSumExp(self, ctx):
        return self.visitChildren(ctx)



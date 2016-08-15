from __future__ import print_function

import sys
from antlr4 import *
from NWChemTCELexer import NWChemTCELexer
from NWChemTCEParser import NWChemTCEParser
from antlr4.tree.Tree import TerminalNodeImpl


codegen = ""


def printuc(s):
    s = str(s)
    asciidata = s
    try:
        u8 = s.decode('utf-8')
        asciidata = u8.encode("ascii", "ignore")
    except UnicodeError:
        pass

    return asciidata

def printres(s):
    global  codegen
    codegen += printuc(s)
    #print(printuc(s))


def printresws(s):
    global codegen
    codegen += " " + printuc(s) + " "
    #print(" " + printuc(s), end=" ")

def printws():
    global codegen
    codegen += " "
    #print(" ", end="")


# This class defines a complete generic visitor for a parse tree produced by NWChemTCEParser.


labelcount_io = 0
labelcount_ia = dict()
namemap = dict()
array_decls = []
#label_prefix = 't1' # label suffix comes from file name ?
lhsanames = dict()
intermediate_decls = dict()
scopes = dict()


class NWChemTCEVisitor(ParseTreeVisitor):

    def __init__(self, x='tce', y='t'):
        self.function_prefix = x
        self.label_prefix = y

    def getCode(self):
        global codegen, intermediate_decls
        return [codegen, intermediate_decls]

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
        global labelcount_io, labelcount_ia, lhsanames, intermediate_decls, scopes
        #self.visitArray_reference(ctx.children[0])
        it = self.print_index_list(((ctx.children[0]).children[2]))
        ilist = it[1]
        lhs_array_name = str((ctx.children[0]).children[0])
        #printresws(lhs_array_name)
        label = ""
        io_flag = False

        if (lhs_array_name[0] != 'i'):
            printres("ARRAY NAME HAS TO START WITH an I \n")
            sys.exit(1)

        ino = int(lhs_array_name[1:])
        indentl = len(self.label_prefix+"_1")+8
        if (lhs_array_name == 'i0'):
            io_flag = True
            labelcount_io = labelcount_io + 1
            label = self.label_prefix + "_" + str(labelcount_io)
            for lname in labelcount_ia.keys():
                labelcount_ia[lname] = 0

        else:
            if lhs_array_name not in labelcount_ia.keys():
                labelcount_ia[lhs_array_name] = 1
            else: labelcount_ia[lhs_array_name] = labelcount_ia[lhs_array_name] + 1

            label = self.label_prefix
            for i in range(1,ino+1):
                label += "_" + str(labelcount_io+1)

            label +=  "_" + str(labelcount_ia[lhs_array_name])
            #labelcount_ia[lhs_array_name] = labelcount_ia[lhs_array_name] + 1

            if lhs_array_name not in lhsanames.keys():
                lhsanames[lhs_array_name] = label
                lhs_array_name = label
            else: lhs_array_name = lhsanames[lhs_array_name]

        printres(label + ":".ljust(indentl-len(label)) + lhs_array_name)
        printres("[")
        printres(ilist)
        printres("]")

        if lhs_array_name not in intermediate_decls.keys():
            il = []
            for index in ilist.split(","):
                if (index[0] == 'h'): il.append('O')
                elif (index[0] == 'p'): il.append('V')
                else: printres("arr index can only start with either p or h\n")

            ilen = len(il)
            intermediate_decls[lhs_array_name] = "[" + ",".join(il[0:ilen/2]) + "][" + ",".join(il[ilen/2:ilen]) + "];"

        # print rhs now
        printws()
        self.visitAssignment_operator(ctx.children[1])
        printws()
        self.visitPtype(ctx.children[2])
        printres("\n")

        # for lnames in intermediate_decls.keys():
        #     printresws("array " + lnames + intermediate_decls[lnames] + "\n")

        lhs_array_name = str((ctx.children[0]).children[0])
        getciv = int(lhs_array_name[1:])
        if io_flag:
            lhsanames.clear()
            scopes.clear()
            return
        if lhs_array_name not in scopes.keys():
            scopes[lhs_array_name] = "i" + str(getciv - 1)
        nciv = getciv + 1
        nextk = "i" + str(getciv + 1)
        if nextk in scopes.keys():
            del scopes[nextk]
            del lhsanames[nextk]
            while True:
                nciv += 1
                nextk = "i" + str(nciv)
                if nextk in scopes.keys():
                    del scopes[nextk]
                    del lhsanames[nextk]
                else:
                    break
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
        global lhsanames
        if isinstance(ctx,NWChemTCEParser.Array_referenceContext):
            aname = str(ctx.children[0]) #print arrayname
            it = self.print_index_list((ctx.children[2])) #print indices
            atype = it[0]
            ilist = it[1]

            arrname = aname
            if(aname[0] == 'i'):
                if aname in lhsanames.keys():
                    arrname = lhsanames[aname]
                    printres(arrname)
            elif aname[0] == 'f' or aname[0] == 'v': printres(arrname)
            else: printres(arrname + "_" + atype)
            printres("[")
            printres(ilist)
            printres("]")
        #return self.visitChildren(ctx)



    def print_index_list(self, idxlist):
        il = ""
        type = ""
        for idx in idxlist.children:
            index = str(idx.children[0])
            if index[-1] == "*": continue
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

    def __init__(self, maxi):
        self.maxi = maxi


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

        if lhs_array_name == 'i0' and 'i0' not in uniqArrDecls.keys():
            atype = self.get_array_type((ctx.children[0]).children[2])
            atypel = atype.split(",")
            al = len(atypel)

            upper = atypel[0:al / 2]
            lower = atypel[al / 2:al]

            newup = []
            newlow = []
            for u in range(0, len(upper)):
                if upper[u][-1] != "*":  newup.append(upper[u])

            for u in range(0, len(lower)):
                if lower[u][-1] != "*": newlow.append(lower[u])

            upper = ",".join(newup)
            lower = ",".join(newlow)

            io_decl = "array i0" + "[" + upper + "]" + "[" + lower + "];"
            uniqArrDecls[lhs_array_name] = lhs_array_name
            array_decls.append(io_decl)


        stmt_seen.append(lhs_array_name)
        if lhs_array_name in stmt_seen:
            if lhs_array_name in stmt_postpone.keys():
                lano = int(lhs_array_name[1:])
                for si in reversed(range(lano,self.maxi+1)):
                    laname = "i" + str(si)
                    if laname in stmt_postpone.keys():
                        addExec = stmt_postpone[laname]
                        exec_order.append(addExec)
                        del stmt_postpone[laname]
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

                atype = it

                atypel = atype.split(",")
                al = len(atypel)
                #if aname[0] == 'f' or aname[0] == 'v': atype = 'N' * al

                upper = atypel[0:al/2]
                lower = atypel[al/2:al]

                newup = []
                newlow = []
                for u in range(0,len(upper)):
                    if upper[u][-1] != "*":
                        if aname[0] == 'f' or aname[0] == 'v': newup.append('N')
                        else: newup.append(upper[u])

                for u in range(0,len(lower)):
                    if lower[u][-1] != "*":
                        if aname[0] == 'f' or aname[0] == 'v': newlow.append('N')
                        else: newlow.append(lower[u])

                upper = ",".join(newup)
                lower = ",".join(newlow)

                newat = []
                for at in range(0,len(atypel)):
                    if atypel[at][-1] != "*": newat.append(atypel[at])

                #if aname[0]=='f' or aname[0] == 'v': atype = 'N'*len(it)

                # al = len(atype)
                # upper = atype[0:al/2]
                # lower = atype[al/2:al]
                # upper = ",".join(upper)
                # lower = ",".join(lower)

                renameArr = aname + "_" + ("".join(newat)).lower()
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
                if index[-1] == "*": type += 'O*,'
                else:
                    type += 'O,'
                    hind = max(hind,int(index[1:]))
            elif (index[0] == 'p'):
                if index[-1] == "*": type += 'V*,'
                else:
                    type += 'V,'
                    pind = max(pind, int(index[1:]))
            else:
                printres("arr index can only start with either p or h\n")

        return type[:-1]
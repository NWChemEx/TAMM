# Generated from /home/panyala/EclipseWS/workspacePTP/tamm/tamm_parser/SoInput/SoTCE.g4 by ANTLR 4.5.3
# encoding: utf-8
from __future__ import print_function
from antlr4 import *
from io import StringIO

def serializedATN():
    with StringIO() as buf:
        buf.write(u"\3\u0430\ud6d1\u8206\uad2d\u4417\uaef1\u8d80\uaadd\3")
        buf.write(u"\21U\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7")
        buf.write(u"\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\3\2\3\2\3\3\3\3\3")
        buf.write(u"\4\3\4\3\5\3\5\5\5\37\n\5\3\6\6\6\"\n\6\r\6\16\6#\3\7")
        buf.write(u"\3\7\6\7(\n\7\r\7\16\7)\3\b\3\b\5\b.\n\b\3\t\3\t\3\t")
        buf.write(u"\3\t\5\t\64\n\t\3\t\3\t\3\t\6\t9\n\t\r\t\16\t:\5\t=\n")
        buf.write(u"\t\3\n\3\n\5\nA\n\n\3\n\7\nD\n\n\f\n\16\nG\13\n\3\n\3")
        buf.write(u"\n\3\13\3\13\3\13\7\13N\n\13\f\13\16\13Q\13\13\3\13\3")
        buf.write(u"\13\3\13\2\2\f\2\4\6\b\n\f\16\20\22\24\2\4\3\2\16\17")
        buf.write(u"\3\2\4\5T\2\26\3\2\2\2\4\30\3\2\2\2\6\32\3\2\2\2\b\36")
        buf.write(u"\3\2\2\2\n!\3\2\2\2\f%\3\2\2\2\16-\3\2\2\2\20<\3\2\2")
        buf.write(u"\2\22>\3\2\2\2\24J\3\2\2\2\26\27\t\2\2\2\27\3\3\2\2\2")
        buf.write(u"\30\31\t\3\2\2\31\5\3\2\2\2\32\33\5\b\5\2\33\7\3\2\2")
        buf.write(u"\2\34\37\7\2\2\3\35\37\5\n\6\2\36\34\3\2\2\2\36\35\3")
        buf.write(u"\2\2\2\37\t\3\2\2\2 \"\5\f\7\2! \3\2\2\2\"#\3\2\2\2#")
        buf.write(u"!\3\2\2\2#$\3\2\2\2$\13\3\2\2\2%\'\7\t\2\2&(\5\16\b\2")
        buf.write(u"\'&\3\2\2\2()\3\2\2\2)\'\3\2\2\2)*\3\2\2\2*\r\3\2\2\2")
        buf.write(u"+.\5\20\t\2,.\7\n\2\2-+\3\2\2\2-,\3\2\2\2.\17\3\2\2\2")
        buf.write(u"/\63\7\6\2\2\60\64\5\24\13\2\61\62\7\f\2\2\62\64\5\22")
        buf.write(u"\n\2\63\60\3\2\2\2\63\61\3\2\2\2\64=\3\2\2\2\65\66\5")
        buf.write(u"\4\3\2\66\67\5\2\2\2\679\3\2\2\28\65\3\2\2\29:\3\2\2")
        buf.write(u"\2:8\3\2\2\2:;\3\2\2\2;=\3\2\2\2</\3\2\2\2<8\3\2\2\2")
        buf.write(u"=\21\3\2\2\2>E\7\7\2\2?A\7\13\2\2@?\3\2\2\2@A\3\2\2\2")
        buf.write(u"AB\3\2\2\2BD\7\r\2\2C@\3\2\2\2DG\3\2\2\2EC\3\2\2\2EF")
        buf.write(u"\3\2\2\2FH\3\2\2\2GE\3\2\2\2HI\7\b\2\2I\23\3\2\2\2JK")
        buf.write(u"\7\3\2\2KO\7\7\2\2LN\7\r\2\2ML\3\2\2\2NQ\3\2\2\2OM\3")
        buf.write(u"\2\2\2OP\3\2\2\2PR\3\2\2\2QO\3\2\2\2RS\7\b\2\2S\25\3")
        buf.write(u"\2\2\2\f\36#)-\63:<@EO")
        return buf.getvalue()


class SoTCEParser ( Parser ):

    grammarFileName = "SoTCE.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ u"<INVALID>", u"'Sum'", u"'+'", u"'-'", u"'*'", u"'('", 
                     u"')'", u"'['", u"']'", u"'=>'" ]

    symbolicNames = [ u"<INVALID>", u"SUM", u"PLUS", u"MINUS", u"TIMES", 
                      u"LPAREN", u"RPAREN", u"LBRACKET", u"RBRACKET", u"SYMOP", 
                      u"ArrType", u"Etype", u"ICONST", u"FCONST", u"COMMENT", 
                      u"WS" ]

    RULE_numerical_constant = 0
    RULE_plusORminus = 1
    RULE_translation_unit = 2
    RULE_compound_element_list_opt = 3
    RULE_compound_element_list = 4
    RULE_factors = 5
    RULE_factors_opt = 6
    RULE_ptype = 7
    RULE_arrDims = 8
    RULE_sumExp = 9

    ruleNames =  [ u"numerical_constant", u"plusORminus", u"translation_unit", 
                   u"compound_element_list_opt", u"compound_element_list", 
                   u"factors", u"factors_opt", u"ptype", u"arrDims", u"sumExp" ]

    EOF = Token.EOF
    SUM=1
    PLUS=2
    MINUS=3
    TIMES=4
    LPAREN=5
    RPAREN=6
    LBRACKET=7
    RBRACKET=8
    SYMOP=9
    ArrType=10
    Etype=11
    ICONST=12
    FCONST=13
    COMMENT=14
    WS=15

    def __init__(self, input):
        super(SoTCEParser, self).__init__(input)
        self.checkVersion("4.5.3")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None



    class Numerical_constantContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(SoTCEParser.Numerical_constantContext, self).__init__(parent, invokingState)
            self.parser = parser

        def ICONST(self):
            return self.getToken(SoTCEParser.ICONST, 0)

        def FCONST(self):
            return self.getToken(SoTCEParser.FCONST, 0)

        def getRuleIndex(self):
            return SoTCEParser.RULE_numerical_constant

        def enterRule(self, listener):
            if hasattr(listener, "enterNumerical_constant"):
                listener.enterNumerical_constant(self)

        def exitRule(self, listener):
            if hasattr(listener, "exitNumerical_constant"):
                listener.exitNumerical_constant(self)

        def accept(self, visitor):
            if hasattr(visitor, "visitNumerical_constant"):
                return visitor.visitNumerical_constant(self)
            else:
                return visitor.visitChildren(self)




    def numerical_constant(self):

        localctx = SoTCEParser.Numerical_constantContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_numerical_constant)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 20
            _la = self._input.LA(1)
            if not(_la==SoTCEParser.ICONST or _la==SoTCEParser.FCONST):
                self._errHandler.recoverInline(self)
            else:
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PlusORminusContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(SoTCEParser.PlusORminusContext, self).__init__(parent, invokingState)
            self.parser = parser
            self.op = None # Token

        def PLUS(self):
            return self.getToken(SoTCEParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(SoTCEParser.MINUS, 0)

        def getRuleIndex(self):
            return SoTCEParser.RULE_plusORminus

        def enterRule(self, listener):
            if hasattr(listener, "enterPlusORminus"):
                listener.enterPlusORminus(self)

        def exitRule(self, listener):
            if hasattr(listener, "exitPlusORminus"):
                listener.exitPlusORminus(self)

        def accept(self, visitor):
            if hasattr(visitor, "visitPlusORminus"):
                return visitor.visitPlusORminus(self)
            else:
                return visitor.visitChildren(self)




    def plusORminus(self):

        localctx = SoTCEParser.PlusORminusContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_plusORminus)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 22
            localctx.op = self._input.LT(1)
            _la = self._input.LA(1)
            if not(_la==SoTCEParser.PLUS or _la==SoTCEParser.MINUS):
                localctx.op = self._errHandler.recoverInline(self)
            else:
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Translation_unitContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(SoTCEParser.Translation_unitContext, self).__init__(parent, invokingState)
            self.parser = parser

        def compound_element_list_opt(self):
            return self.getTypedRuleContext(SoTCEParser.Compound_element_list_optContext,0)


        def getRuleIndex(self):
            return SoTCEParser.RULE_translation_unit

        def enterRule(self, listener):
            if hasattr(listener, "enterTranslation_unit"):
                listener.enterTranslation_unit(self)

        def exitRule(self, listener):
            if hasattr(listener, "exitTranslation_unit"):
                listener.exitTranslation_unit(self)

        def accept(self, visitor):
            if hasattr(visitor, "visitTranslation_unit"):
                return visitor.visitTranslation_unit(self)
            else:
                return visitor.visitChildren(self)




    def translation_unit(self):

        localctx = SoTCEParser.Translation_unitContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_translation_unit)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 24
            self.compound_element_list_opt()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Compound_element_list_optContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(SoTCEParser.Compound_element_list_optContext, self).__init__(parent, invokingState)
            self.parser = parser

        def EOF(self):
            return self.getToken(SoTCEParser.EOF, 0)

        def compound_element_list(self):
            return self.getTypedRuleContext(SoTCEParser.Compound_element_listContext,0)


        def getRuleIndex(self):
            return SoTCEParser.RULE_compound_element_list_opt

        def enterRule(self, listener):
            if hasattr(listener, "enterCompound_element_list_opt"):
                listener.enterCompound_element_list_opt(self)

        def exitRule(self, listener):
            if hasattr(listener, "exitCompound_element_list_opt"):
                listener.exitCompound_element_list_opt(self)

        def accept(self, visitor):
            if hasattr(visitor, "visitCompound_element_list_opt"):
                return visitor.visitCompound_element_list_opt(self)
            else:
                return visitor.visitChildren(self)




    def compound_element_list_opt(self):

        localctx = SoTCEParser.Compound_element_list_optContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_compound_element_list_opt)
        try:
            self.state = 28
            token = self._input.LA(1)
            if token in [SoTCEParser.EOF]:
                self.enterOuterAlt(localctx, 1)
                self.state = 26
                self.match(SoTCEParser.EOF)

            elif token in [SoTCEParser.LBRACKET]:
                self.enterOuterAlt(localctx, 2)
                self.state = 27
                self.compound_element_list()

            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Compound_element_listContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(SoTCEParser.Compound_element_listContext, self).__init__(parent, invokingState)
            self.parser = parser

        def factors(self, i=None):
            if i is None:
                return self.getTypedRuleContexts(SoTCEParser.FactorsContext)
            else:
                return self.getTypedRuleContext(SoTCEParser.FactorsContext,i)


        def getRuleIndex(self):
            return SoTCEParser.RULE_compound_element_list

        def enterRule(self, listener):
            if hasattr(listener, "enterCompound_element_list"):
                listener.enterCompound_element_list(self)

        def exitRule(self, listener):
            if hasattr(listener, "exitCompound_element_list"):
                listener.exitCompound_element_list(self)

        def accept(self, visitor):
            if hasattr(visitor, "visitCompound_element_list"):
                return visitor.visitCompound_element_list(self)
            else:
                return visitor.visitChildren(self)




    def compound_element_list(self):

        localctx = SoTCEParser.Compound_element_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_compound_element_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 31 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 30
                self.factors()
                self.state = 33 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==SoTCEParser.LBRACKET):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class FactorsContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(SoTCEParser.FactorsContext, self).__init__(parent, invokingState)
            self.parser = parser

        def LBRACKET(self):
            return self.getToken(SoTCEParser.LBRACKET, 0)

        def factors_opt(self, i=None):
            if i is None:
                return self.getTypedRuleContexts(SoTCEParser.Factors_optContext)
            else:
                return self.getTypedRuleContext(SoTCEParser.Factors_optContext,i)


        def getRuleIndex(self):
            return SoTCEParser.RULE_factors

        def enterRule(self, listener):
            if hasattr(listener, "enterFactors"):
                listener.enterFactors(self)

        def exitRule(self, listener):
            if hasattr(listener, "exitFactors"):
                listener.exitFactors(self)

        def accept(self, visitor):
            if hasattr(visitor, "visitFactors"):
                return visitor.visitFactors(self)
            else:
                return visitor.visitChildren(self)




    def factors(self):

        localctx = SoTCEParser.FactorsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_factors)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 35
            self.match(SoTCEParser.LBRACKET)
            self.state = 37 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 36
                self.factors_opt()
                self.state = 39 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SoTCEParser.PLUS) | (1 << SoTCEParser.MINUS) | (1 << SoTCEParser.TIMES) | (1 << SoTCEParser.RBRACKET))) != 0)):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Factors_optContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(SoTCEParser.Factors_optContext, self).__init__(parent, invokingState)
            self.parser = parser

        def ptype(self):
            return self.getTypedRuleContext(SoTCEParser.PtypeContext,0)


        def RBRACKET(self):
            return self.getToken(SoTCEParser.RBRACKET, 0)

        def getRuleIndex(self):
            return SoTCEParser.RULE_factors_opt

        def enterRule(self, listener):
            if hasattr(listener, "enterFactors_opt"):
                listener.enterFactors_opt(self)

        def exitRule(self, listener):
            if hasattr(listener, "exitFactors_opt"):
                listener.exitFactors_opt(self)

        def accept(self, visitor):
            if hasattr(visitor, "visitFactors_opt"):
                return visitor.visitFactors_opt(self)
            else:
                return visitor.visitChildren(self)




    def factors_opt(self):

        localctx = SoTCEParser.Factors_optContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_factors_opt)
        try:
            self.state = 43
            token = self._input.LA(1)
            if token in [SoTCEParser.PLUS, SoTCEParser.MINUS, SoTCEParser.TIMES]:
                self.enterOuterAlt(localctx, 1)
                self.state = 41
                self.ptype()

            elif token in [SoTCEParser.RBRACKET]:
                self.enterOuterAlt(localctx, 2)
                self.state = 42
                self.match(SoTCEParser.RBRACKET)

            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class PtypeContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(SoTCEParser.PtypeContext, self).__init__(parent, invokingState)
            self.parser = parser

        def TIMES(self):
            return self.getToken(SoTCEParser.TIMES, 0)

        def sumExp(self):
            return self.getTypedRuleContext(SoTCEParser.SumExpContext,0)


        def ArrType(self):
            return self.getToken(SoTCEParser.ArrType, 0)

        def arrDims(self):
            return self.getTypedRuleContext(SoTCEParser.ArrDimsContext,0)


        def plusORminus(self, i=None):
            if i is None:
                return self.getTypedRuleContexts(SoTCEParser.PlusORminusContext)
            else:
                return self.getTypedRuleContext(SoTCEParser.PlusORminusContext,i)


        def numerical_constant(self, i=None):
            if i is None:
                return self.getTypedRuleContexts(SoTCEParser.Numerical_constantContext)
            else:
                return self.getTypedRuleContext(SoTCEParser.Numerical_constantContext,i)


        def getRuleIndex(self):
            return SoTCEParser.RULE_ptype

        def enterRule(self, listener):
            if hasattr(listener, "enterPtype"):
                listener.enterPtype(self)

        def exitRule(self, listener):
            if hasattr(listener, "exitPtype"):
                listener.exitPtype(self)

        def accept(self, visitor):
            if hasattr(visitor, "visitPtype"):
                return visitor.visitPtype(self)
            else:
                return visitor.visitChildren(self)




    def ptype(self):

        localctx = SoTCEParser.PtypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_ptype)
        try:
            self.state = 58
            token = self._input.LA(1)
            if token in [SoTCEParser.TIMES]:
                self.enterOuterAlt(localctx, 1)
                self.state = 45
                self.match(SoTCEParser.TIMES)
                self.state = 49
                token = self._input.LA(1)
                if token in [SoTCEParser.SUM]:
                    self.state = 46
                    self.sumExp()

                elif token in [SoTCEParser.ArrType]:
                    self.state = 47
                    self.match(SoTCEParser.ArrType)
                    self.state = 48
                    self.arrDims()

                else:
                    raise NoViableAltException(self)


            elif token in [SoTCEParser.PLUS, SoTCEParser.MINUS]:
                self.enterOuterAlt(localctx, 2)
                self.state = 54 
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 51
                        self.plusORminus()
                        self.state = 52
                        self.numerical_constant()

                    else:
                        raise NoViableAltException(self)
                    self.state = 56 
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,5,self._ctx)


            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ArrDimsContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(SoTCEParser.ArrDimsContext, self).__init__(parent, invokingState)
            self.parser = parser

        def LPAREN(self):
            return self.getToken(SoTCEParser.LPAREN, 0)

        def RPAREN(self):
            return self.getToken(SoTCEParser.RPAREN, 0)

        def Etype(self, i=None):
            if i is None:
                return self.getTokens(SoTCEParser.Etype)
            else:
                return self.getToken(SoTCEParser.Etype, i)

        def SYMOP(self, i=None):
            if i is None:
                return self.getTokens(SoTCEParser.SYMOP)
            else:
                return self.getToken(SoTCEParser.SYMOP, i)

        def getRuleIndex(self):
            return SoTCEParser.RULE_arrDims

        def enterRule(self, listener):
            if hasattr(listener, "enterArrDims"):
                listener.enterArrDims(self)

        def exitRule(self, listener):
            if hasattr(listener, "exitArrDims"):
                listener.exitArrDims(self)

        def accept(self, visitor):
            if hasattr(visitor, "visitArrDims"):
                return visitor.visitArrDims(self)
            else:
                return visitor.visitChildren(self)




    def arrDims(self):

        localctx = SoTCEParser.ArrDimsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_arrDims)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 60
            self.match(SoTCEParser.LPAREN)
            self.state = 67
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==SoTCEParser.SYMOP or _la==SoTCEParser.Etype:
                self.state = 62
                _la = self._input.LA(1)
                if _la==SoTCEParser.SYMOP:
                    self.state = 61
                    self.match(SoTCEParser.SYMOP)


                self.state = 64
                self.match(SoTCEParser.Etype)
                self.state = 69
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 70
            self.match(SoTCEParser.RPAREN)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class SumExpContext(ParserRuleContext):

        def __init__(self, parser, parent=None, invokingState=-1):
            super(SoTCEParser.SumExpContext, self).__init__(parent, invokingState)
            self.parser = parser

        def SUM(self):
            return self.getToken(SoTCEParser.SUM, 0)

        def LPAREN(self):
            return self.getToken(SoTCEParser.LPAREN, 0)

        def RPAREN(self):
            return self.getToken(SoTCEParser.RPAREN, 0)

        def Etype(self, i=None):
            if i is None:
                return self.getTokens(SoTCEParser.Etype)
            else:
                return self.getToken(SoTCEParser.Etype, i)

        def getRuleIndex(self):
            return SoTCEParser.RULE_sumExp

        def enterRule(self, listener):
            if hasattr(listener, "enterSumExp"):
                listener.enterSumExp(self)

        def exitRule(self, listener):
            if hasattr(listener, "exitSumExp"):
                listener.exitSumExp(self)

        def accept(self, visitor):
            if hasattr(visitor, "visitSumExp"):
                return visitor.visitSumExp(self)
            else:
                return visitor.visitChildren(self)




    def sumExp(self):

        localctx = SoTCEParser.SumExpContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_sumExp)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 72
            self.match(SoTCEParser.SUM)
            self.state = 73
            self.match(SoTCEParser.LPAREN)
            self.state = 77
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==SoTCEParser.Etype:
                self.state = 74
                self.match(SoTCEParser.Etype)
                self.state = 79
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 80
            self.match(SoTCEParser.RPAREN)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx






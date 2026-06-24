# Generated from Dml.g4 by ANTLR 4.13.0
from antlr4 import *
if "." in __name__:
    from .DmlParser import DmlParser
else:
    from DmlParser import DmlParser

# This class defines a complete generic visitor for a parse tree produced by DmlParser.

class DmlVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by DmlParser#programroot.
    def visitProgramroot(self, ctx:DmlParser.ProgramrootContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ImportStatement.
    def visitImportStatement(self, ctx:DmlParser.ImportStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#PathStatement.
    def visitPathStatement(self, ctx:DmlParser.PathStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#FunctionCallAssignmentStatement.
    def visitFunctionCallAssignmentStatement(self, ctx:DmlParser.FunctionCallAssignmentStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#FunctionCallMultiAssignmentStatement.
    def visitFunctionCallMultiAssignmentStatement(self, ctx:DmlParser.FunctionCallMultiAssignmentStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#IfdefAssignmentStatement.
    def visitIfdefAssignmentStatement(self, ctx:DmlParser.IfdefAssignmentStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#AssignmentStatement.
    def visitAssignmentStatement(self, ctx:DmlParser.AssignmentStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#AccumulatorAssignmentStatement.
    def visitAccumulatorAssignmentStatement(self, ctx:DmlParser.AccumulatorAssignmentStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#IfStatement.
    def visitIfStatement(self, ctx:DmlParser.IfStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ForStatement.
    def visitForStatement(self, ctx:DmlParser.ForStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ParForStatement.
    def visitParForStatement(self, ctx:DmlParser.ParForStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#WhileStatement.
    def visitWhileStatement(self, ctx:DmlParser.WhileStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#IterablePredicateColonExpression.
    def visitIterablePredicateColonExpression(self, ctx:DmlParser.IterablePredicateColonExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#IterablePredicateSeqExpression.
    def visitIterablePredicateSeqExpression(self, ctx:DmlParser.IterablePredicateSeqExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#InternalFunctionDefExpression.
    def visitInternalFunctionDefExpression(self, ctx:DmlParser.InternalFunctionDefExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ExternalFunctionDefExpression.
    def visitExternalFunctionDefExpression(self, ctx:DmlParser.ExternalFunctionDefExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#IndexedExpression.
    def visitIndexedExpression(self, ctx:DmlParser.IndexedExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#SimpleDataIdentifierExpression.
    def visitSimpleDataIdentifierExpression(self, ctx:DmlParser.SimpleDataIdentifierExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#CommandlineParamExpression.
    def visitCommandlineParamExpression(self, ctx:DmlParser.CommandlineParamExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#CommandlinePositionExpression.
    def visitCommandlinePositionExpression(self, ctx:DmlParser.CommandlinePositionExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ModIntDivExpression.
    def visitModIntDivExpression(self, ctx:DmlParser.ModIntDivExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#RelationalExpression.
    def visitRelationalExpression(self, ctx:DmlParser.RelationalExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#BooleanNotExpression.
    def visitBooleanNotExpression(self, ctx:DmlParser.BooleanNotExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#PowerExpression.
    def visitPowerExpression(self, ctx:DmlParser.PowerExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#BuiltinFunctionExpression.
    def visitBuiltinFunctionExpression(self, ctx:DmlParser.BuiltinFunctionExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ConstIntIdExpression.
    def visitConstIntIdExpression(self, ctx:DmlParser.ConstIntIdExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#AtomicExpression.
    def visitAtomicExpression(self, ctx:DmlParser.AtomicExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ConstStringIdExpression.
    def visitConstStringIdExpression(self, ctx:DmlParser.ConstStringIdExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ConstTrueExpression.
    def visitConstTrueExpression(self, ctx:DmlParser.ConstTrueExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#UnaryExpression.
    def visitUnaryExpression(self, ctx:DmlParser.UnaryExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#MultDivExpression.
    def visitMultDivExpression(self, ctx:DmlParser.MultDivExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ConstFalseExpression.
    def visitConstFalseExpression(self, ctx:DmlParser.ConstFalseExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#DataIdExpression.
    def visitDataIdExpression(self, ctx:DmlParser.DataIdExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#AddSubExpression.
    def visitAddSubExpression(self, ctx:DmlParser.AddSubExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ConstDoubleIdExpression.
    def visitConstDoubleIdExpression(self, ctx:DmlParser.ConstDoubleIdExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#MatrixMulExpression.
    def visitMatrixMulExpression(self, ctx:DmlParser.MatrixMulExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#MultiIdExpression.
    def visitMultiIdExpression(self, ctx:DmlParser.MultiIdExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#BooleanAndExpression.
    def visitBooleanAndExpression(self, ctx:DmlParser.BooleanAndExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#BooleanOrExpression.
    def visitBooleanOrExpression(self, ctx:DmlParser.BooleanOrExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#typedArgNoAssign.
    def visitTypedArgNoAssign(self, ctx:DmlParser.TypedArgNoAssignContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#typedArgAssign.
    def visitTypedArgAssign(self, ctx:DmlParser.TypedArgAssignContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#parameterizedExpression.
    def visitParameterizedExpression(self, ctx:DmlParser.ParameterizedExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#strictParameterizedExpression.
    def visitStrictParameterizedExpression(self, ctx:DmlParser.StrictParameterizedExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#strictParameterizedKeyValueString.
    def visitStrictParameterizedKeyValueString(self, ctx:DmlParser.StrictParameterizedKeyValueStringContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#ml_type.
    def visitMl_type(self, ctx:DmlParser.Ml_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#valueType.
    def visitValueType(self, ctx:DmlParser.ValueTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by DmlParser#MatrixDataTypeCheck.
    def visitMatrixDataTypeCheck(self, ctx:DmlParser.MatrixDataTypeCheckContext):
        return self.visitChildren(ctx)



del DmlParser

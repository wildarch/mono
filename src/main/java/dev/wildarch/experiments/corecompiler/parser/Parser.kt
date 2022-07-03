package dev.wildarch.experiments.corecompiler.parser

import dev.wildarch.corecompiler.CoreBaseVisitor
import dev.wildarch.corecompiler.CoreLexer
import dev.wildarch.corecompiler.CoreParser
import dev.wildarch.experiments.corecompiler.syntax.*
import org.antlr.v4.runtime.*
import org.antlr.v4.runtime.tree.ParseTree

fun parse(source: String) : Program {
    val lexer = CoreLexer(CharStreams.fromString(source))
    val tokens = CommonTokenStream(lexer)
    val parser = CoreParser(tokens)
    parser.removeErrorListeners()
    val errorListener = ErrorListener()
    parser.addErrorListener(errorListener)
    val tree: ParseTree = parser.program()
    if (errorListener.errors.isNotEmpty()) {
        error("${errorListener.errors.size} parse errors: ${errorListener.errors.joinToString(separator = "\n")}")
    }
    val astParser = Parser()
    return astParser.visit(tree) as Program
}

class Parser : CoreBaseVisitor<AstNode>() {
    override fun visitProgram(ctx: CoreParser.ProgramContext): Program {
        return Program(ctx.sc().map(::visitSc).toList())
    }

    override fun visitSc(ctx: CoreParser.ScContext): ScDefn {
        return ScDefn(
            ctx.`var`(0).text,
            ctx.`var`().subList(1, ctx.`var`().size).map { it.text },
            visit(ctx.expr()) as Expr
        )
    }

    override fun visitExprVar(ctx: CoreParser.ExprVarContext): Var {
        return Var(ctx.text)
    }

    override fun visitExprNum(ctx: CoreParser.ExprNumContext): Num {
        return Num(ctx.text.toInt())
    }

    override fun visitExprConstr(ctx: CoreParser.ExprConstrContext): Constr {
        val tag = ctx.num(0).text.toInt()
        val arity = ctx.num(1).text.toInt()
        return Constr(tag, arity)
    }

    override fun visitExprParen(ctx: CoreParser.ExprParenContext): Expr {
        return visit(ctx.expr()) as Expr
    }
    override fun visitExprApp(ctx: CoreParser.ExprAppContext): Ap {
        return Ap(
            visit(ctx.expr(0)) as Expr,
            visit(ctx.expr(1)) as Expr
        )
    }

    override fun visitExprMulDiv(ctx: CoreParser.ExprMulDivContext): BinOp {
        val lhs = visit(ctx.expr(0)) as Expr
        val rhs = visit(ctx.expr(1)) as Expr
        val op = when (ctx.op.type) {
            CoreParser.MUL -> Operator.MUL
            CoreParser.DIV -> Operator.DIV
            else -> error("unknown op type")
        }
        return BinOp(lhs, op, rhs)
    }

    override fun visitExprAddSub(ctx: CoreParser.ExprAddSubContext): BinOp {
        val lhs = visit(ctx.expr(0)) as Expr
        val rhs = visit(ctx.expr(1)) as Expr
        val op = when (ctx.op.type) {
            CoreParser.ADD -> Operator.ADD
            CoreParser.SUB -> Operator.SUB
            else -> error("unknown op type")
        }
        return BinOp(lhs, op, rhs)
    }

    override fun visitExprCompare(ctx: CoreParser.ExprCompareContext): BinOp {
        val lhs = visit(ctx.expr(0)) as Expr
        val rhs = visit(ctx.expr(1)) as Expr
        val op = when (ctx.op.type) {
            CoreParser.EQ -> Operator.EQ
            CoreParser.NEQ -> Operator.NEQ
            CoreParser.GT -> Operator.GT
            CoreParser.GTE -> Operator.GTE
            CoreParser.LT -> Operator.LT
            CoreParser.LTE -> Operator.LTE
            else -> error("unknown op type")
        }
        return BinOp(lhs, op, rhs)
    }

    override fun visitExprAnd(ctx: CoreParser.ExprAndContext): BinOp {
        val lhs = visit(ctx.expr(0)) as Expr
        val rhs = visit(ctx.expr(1)) as Expr

        return BinOp(
            lhs,
            Operator.AND,
            rhs,
        )
    }

    override fun visitExprOr(ctx: CoreParser.ExprOrContext): BinOp {
        val lhs = visit(ctx.expr(0)) as Expr
        val rhs = visit(ctx.expr(1)) as Expr

        return BinOp(
            lhs,
            Operator.OR,
            rhs,
        )
    }

    override fun visitExprLet(ctx: CoreParser.ExprLetContext): Let {
        val isRec = when(ctx.rec.type) {
            CoreParser.LET -> false
            CoreParser.LETREC -> true
            else -> error("unknown let type")
        }

        val defns = ctx.defns().defn().map(::visitDefn)
        val expr = visit(ctx.expr()) as Expr

        return Let(isRec, defns, expr)
    }

    override fun visitDefn(ctx: CoreParser.DefnContext): Def {
        val name = ctx.`var`().text
        val expr = visit(ctx.expr()) as Expr

        return Def(name, expr)
    }

    override fun visitExprCase(ctx: CoreParser.ExprCaseContext): Case {
        val expr = visit(ctx.expr()) as Expr
        val alts = ctx.alts().alt().map(::visitAlt)

        return Case(expr, alts)
    }

    override fun visitAlt(ctx: CoreParser.AltContext): Alter {
        val tag = ctx.num().text.toInt()
        val binds = ctx.`var`().map { it.text }
        val body = visit(ctx.expr()) as Expr

        return Alter(tag, binds, body)
    }

    override fun visitExprLam(ctx: CoreParser.ExprLamContext): Lam {
        val params = ctx.`var`().map { it.text }
        val body = visit(ctx.expr()) as Expr

        return Lam(params, body)
    }
}

class ErrorListener : BaseErrorListener() {
    val errors = mutableListOf<String>()
    override fun syntaxError(
        recognizer: Recognizer<*, *>,
        offendingSymbol: Any,
        line: Int,
        charPositionInLine: Int,
        msg: String,
        e: RecognitionException?
    ) {
        val stack = (recognizer as CoreParser).ruleInvocationStack.reversed()
        errors += "rule stack: $stack\nline $line:$charPositionInLine at $offendingSymbol: $msg"
    }
}
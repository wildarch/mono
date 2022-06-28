package dev.wildarch.experiments.corecompiler

import dev.wildarch.corecompiler.CoreBaseVisitor
import dev.wildarch.corecompiler.CoreParser
import dev.wildarch.experiments.corecompiler.syntax.*

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

    override fun visitExprAddSub(ctx: CoreParser.ExprAddSubContext): Expr {
        val lhs = visit(ctx.expr(0)) as Expr
        val rhs = visit(ctx.expr(1)) as Expr
        // TODO: fix
        return BinOp(lhs, Operator.ADD, rhs)
    }

    override fun visitExprApp(ctx: CoreParser.ExprAppContext): Expr {
        return Ap(
            visit(ctx.expr(0)) as Expr,
            visit(ctx.expr(1)) as Expr
        )
    }

    override fun visitExprVar(ctx: CoreParser.ExprVarContext): Expr {
        return Var(ctx.text)
    }

    override fun visitExprNum(ctx: CoreParser.ExprNumContext): Expr {
        return Num(ctx.text.toInt())
    }

    override fun visitExprParen(ctx: CoreParser.ExprParenContext): Expr {
        return visit(ctx.expr()) as Expr
    }
}
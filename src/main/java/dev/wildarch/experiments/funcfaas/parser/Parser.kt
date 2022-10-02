package dev.wildarch.experiments.funcfaas.parser

import dev.wildarch.experiments.funcfaas.syntax.*
import org.antlr.v4.runtime.*
import org.antlr.v4.runtime.tree.ParseTree

fun parseModule(source: String): Module {
    val lexer = BrimLexer(CharStreams.fromString(source))
    val tokens = CommonTokenStream(lexer)
    val parser = BrimParser(tokens)
    parser.removeErrorListeners()
    val errorListener = ErrorListener()
    parser.addErrorListener(errorListener)
    val tree: ParseTree = parser.module()
    if (errorListener.errors.isNotEmpty()) {
        error("${errorListener.errors.size} parse errors: ${errorListener.errors.joinToString(separator = "\n")}")
    }
    val astParser = Parser()
    return astParser.visit(tree) as Module
}

class Parser : BrimBaseVisitor<AstNode>() {
    override fun visitModule(ctx: BrimParser.ModuleContext): Module {
        // TODO: type defs
        val dataTypes = emptyList<TypeDef>()
        val supercombs = ctx.supercomb().map(::visitSupercomb).toList()
        return Module(dataTypes, supercombs)
    }

    override fun visitSupercomb(ctx: BrimParser.SupercombContext): Supercomb {
        // Name first appears in the type signature
        val name = visitVar(ctx.`var`(0)).name
        // Name appears a second time when defining the implementation
        val nameInImpl = visitVar(ctx.`var`(1)).name
        if (name != nameInImpl) {
            error("name in type signature $name should match the one in the implementation $nameInImpl")
        }

        val type = visit(ctx.type()) as Type
        val params = ctx.`var`().drop(2).map(::visitVar).map { it.name }
        val body = visit(ctx.expr()) as Expr
        return Supercomb(name, type, params, body)
    }

    override fun visitVar(ctx: BrimParser.VarContext): Var {
        return Var(ctx.text)
    }

    override fun visitTypeVar(ctx: BrimParser.TypeVarContext): TypeVar {
        return TypeVar(visitVar(ctx.`var`()).name)
    }

    override fun visitTypeFun(ctx: BrimParser.TypeFunContext): TypeFun {
        return TypeFun(
            arg = visit(ctx.type(0)) as Type,
            ret = visit(ctx.type(1)) as Type,
        )
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
        val stack = (recognizer as BrimParser).ruleInvocationStack.reversed()
        errors += "rule stack: $stack\nline $line:$charPositionInLine at $offendingSymbol: $msg"
    }
}

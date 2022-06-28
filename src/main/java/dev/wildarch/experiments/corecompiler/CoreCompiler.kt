package dev.wildarch.experiments.corecompiler

import dev.wildarch.corecompiler.CoreLexer
import dev.wildarch.corecompiler.CoreParser
import dev.wildarch.experiments.corecompiler.syntax.Program
import org.antlr.v4.runtime.CharStreams
import org.antlr.v4.runtime.CommonTokenStream
import org.antlr.v4.runtime.tree.ParseTree
import org.antlr.v4.runtime.tree.Trees

fun main() {
    val program = "f x = x + (negate x)"
    val lexer = CoreLexer(CharStreams.fromString(program))
    val tokens = CommonTokenStream(lexer)
    val parser = CoreParser(tokens)
    val tree: ParseTree = parser.program()
    println("Tree: " + Trees.toStringTree(tree))

    val astParser = Parser()
    val ast: Program = astParser.visit(tree) as Program
    println("AST: $ast")
}
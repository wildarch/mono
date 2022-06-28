package dev.wildarch.experiments.corecompiler.syntax

data class Program(val defs: List<ScDefn>) : AstNode()

data class ScDefn(val name: String, val params: List<String>, val body: Expr) : AstNode()
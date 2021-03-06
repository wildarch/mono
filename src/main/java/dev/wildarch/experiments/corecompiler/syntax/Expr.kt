package dev.wildarch.experiments.corecompiler.syntax

sealed class AstNode

sealed class Expr : AstNode()

data class Var(val name: String) : Expr()
data class Num(val value: Int) : Expr()
data class Constr(val tag: Int, val arity: Int) : Expr()
data class Ap(val func: Expr, val arg: Expr) : Expr()
data class Let(val isRec: Boolean, val defs: List<Def>, val body: Expr) : Expr()
data class Case(val expr: Expr, val alters: List<Alter>): Expr()
data class Lam(val params: List<String>, val body: Expr): Expr()
data class BinOp(val lhs: Expr, val op: Operator, val rhs: Expr) : Expr()

data class Alter(val tag: Int, val binds: List<String>, val body: Expr) : AstNode()

data class Def(val name: String, val expr: Expr) : AstNode()

enum class Operator {
    ADD,
    SUB,
    MUL,
    DIV,
    EQ,
    NEQ,
    GT,
    GTE,
    LT,
    LTE,
    AND,
    OR,
}

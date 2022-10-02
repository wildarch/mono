package dev.wildarch.experiments.funcfaas.syntax

sealed class AstNode

data class Module(
    val dataTypes: List<TypeDef>,
    val supercombs: List<Supercomb>,
) : AstNode()

data class TypeDef(
    val name: String,
    val constructors: List<Constructor>,
) : AstNode()

data class Constructor(
    val name: String,
    val parameters: List<Type>,
) : AstNode()

sealed class Type : AstNode()

data class TypeVar(val name: String) : Type()
data class TypeFun(val arg: Type, val ret: Type): Type()

data class Supercomb(
    val name: String,
    val type: Type,
    val params: List<String>,
    val body: Expr,
) : AstNode()

sealed class Expr : AstNode()

data class Var(val name: String) : Expr()
data class Num(val value: Int) : Expr()
data class Ap(val func: Expr, val arg: Expr) : Expr()
data class BinOp(val lhs: Expr, val op: Operator, val rhs: Expr) : Expr()

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

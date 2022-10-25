package dev.wildarch.experiments.corecompiler.ski

import dev.wildarch.experiments.corecompiler.syntax.*

fun compile(program: Program): SkState {
    /* TODO
     * - Compile each function to lambda
     * - Compile the body of main, with all function references inlined.
     * - Return only the final compiled body, which is now self-contained
     * - Should match the output of https://crypto.stanford.edu/~blynn/lambda/sk.html
     */

    return SkState(compileC(inlineAll(program)), emptyList())
}

// Inline all function calls, turning the program into one expression
private fun inlineAll(program: Program): Expr {
    val defs = buildMap {
        for (def in program.defs) {
            val expr = if (def.params.isEmpty()) {
                def.body
            } else {
                Lam(def.params, def.body)
            }
            put(def.name, expr)
        }
    }

    return inline(defs, defs["main"] ?: error("no main function"))
}

private fun inline(defs: Map<String, Expr>, main: Expr): Expr {
    return when (main) {
        is dev.wildarch.experiments.corecompiler.syntax.Ap -> Ap(inline(defs, main.func), inline(defs, main.arg))
        is BinOp -> BinOp(inline(defs, main.lhs), main.op, inline(defs, main.rhs))
        //is Case -> Case(inline(defs, main.expr), main.alters.map { inline(defs, it) })
        is Case -> TODO()
        is Constr -> main
        is Lam -> Lam(main.params, inline(defs, main.body))
        is Let -> TODO()
        is Num -> main
        is Var -> defs[main.name] ?: main
    }
}

private fun compileC(expr: Expr): Comb {
    return when (expr) {
        is dev.wildarch.experiments.corecompiler.syntax.Ap -> Ap(compileC(expr.func), compileC(expr.arg))
        is BinOp -> TODO()
        is Case -> TODO()
        is Constr -> TODO()
        is Lam -> {
            var exprC = compileC(expr.body)
            for (param in expr.params.reversed()) {
                exprC = compileA(param, exprC)
            }
            exprC
        }

        is Let -> TODO()
        is Num -> ConstInt(expr.value)
        is Var -> when (expr.name) {
            "I" -> I
            "K" -> K
            "S" -> S
            else -> ConstVar(expr.name)
        }
    }
}

private fun compileA(param: String, expr: Comb): Comb {
    return when (expr) {
        is Ap -> Ap(Ap(S, compileA(param, expr.func)), compileA(param, expr.arg))

        is ConstVar ->
            if (expr.name == param) {
                I
            } else {
                Ap(K, expr)
            }

        else -> Ap(K, expr)
    }
}

fun evaluate(initialState: SkState, maxSteps: Int = 10000): List<SkState> {
    val trace = mutableListOf(initialState)
    var steps = 0
    while (!trace.last().isFinal()) {
        if (steps > maxSteps) {
            error("Did not terminate after $maxSteps")
        }
        trace.add(step(trace.last()))
        steps++;
    }
    return trace
}

private fun step(state: SkState): SkState {
    /*
    return SkState(when(val code = state.code) {
        is Ap -> when (val func = code.func) {
            is Ap -> when (val func2 = func.func) {
                is Ap -> when (func2.func) {
                    S -> TODO()
                    else -> error("Weird Ap")
                }
                K -> func.arg
                else -> error("Weird Ap")
            }
            is ConstInt -> error("Apply to int")
            is ConstVar -> error("Var remaining")
            I -> code.arg
            S -> error("Need a second and third argument")
            K -> error("Need a second argument")
        }
        is ConstInt -> code
        is ConstVar -> code
        I -> TODO()
        K -> TODO()
        S -> TODO()
    })
     */

    return when (val code = state.code) {
        is Ap ->
            SkState(
                code = code.func,
                stack = state.stack + code.arg
            )
        is ConstInt -> error("Cannot reduce int")
        is ConstVar -> error("stray var")
        // I x = x
        I -> SkState(
            code = state.stack.last(),
            stack = state.stack.dropLast(1)
        )
        // K c x = c
        K -> SkState(
            code = state.stack.last(),
            stack = state.stack.dropLast(2)
        )
        S -> TODO("Evaluate argument first")
    }
}

data class SkState(val code: Comb, val stack: List<Comb>) {
    fun isFinal() = code is ConstInt
}

sealed class Comb

data class Ap(val func: Comb, val arg: Comb) : Comb()
data class ConstInt(val n: Int) : Comb()
data class ConstVar(val name: String) : Comb()
object I : Comb()
object S : Comb()
object K : Comb()

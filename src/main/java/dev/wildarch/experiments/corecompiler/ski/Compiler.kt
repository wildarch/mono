package dev.wildarch.experiments.corecompiler.ski

import dev.wildarch.experiments.corecompiler.syntax.*

fun compile(program: Program): SkState {
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

    return inline(defs, defs["main"] ?: error("no main function"), emptyList())
}

private fun inline(defs: Map<String, Expr>, expr: Expr, stack: List<String>): Expr {
    // Check for cycles in inlining
    if (stack.isNotEmpty() && stack.indexOf(stack.last()) != stack.size-1) {
        // Last item found at an earlier position == cycle
        error("Cyclic inlining detected: $stack")
    }

    return when (expr) {
        is dev.wildarch.experiments.corecompiler.syntax.Ap -> Ap(inline(defs, expr.func, stack), inline(defs, expr.arg, stack))
        is BinOp -> BinOp(inline(defs, expr.lhs, stack), expr.op, inline(defs, expr.rhs, stack))
        is Case -> Case(inline(defs, expr.expr, stack), expr.alters.map {it.copy(body = inline(defs, it.body, stack))})
        is Constr -> expr
        is Lam -> Lam(expr.params, inline(defs, expr.body, stack))
        is Let -> {
            /*
             * Transforms
             * ```
             * let
             *      b0 = e0
             *      b1 = e1
             * in
             *      E
             * ```
             *
             * into
             *
             * ```
             * (\b0 b1 . E) e0 e1
             * ```
             *
             */
            //assert(!expr.isRec)
            var lamb: Expr = Lam(expr.defs.map {it.name}, inline(defs, expr.body, stack))
            for (bind in expr.defs) {
                if (expr.isRec) {
                    TODO()
                } else {
                    lamb = Ap(lamb, inline(defs, bind.expr, stack))
                }
            }
            lamb
        }
        is Num -> expr
        is Var -> defs[expr.name]?.let { inline(defs, it, stack + expr.name) } ?: expr
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
    return when (val code = state.code) {
        is Ap ->
            SkState(
                code = code.func,
                stack = state.stack + code.arg
            )
        is ConstInt -> {
            // Continue executing S
            val x = code
            val f = state.stack[state.stack.size-1]
            val g = state.stack[state.stack.size-2]

            return SkState(
                code = Ap(Ap(f, x), Ap(g, x)),
                stack = state.stack.dropLast(2)
            )
        }
        is ConstVar -> error("stray var ${code.name}")
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
        // S f g x = f x (g x)
        S -> {
            val f = state.stack[state.stack.size-1]
            val g = state.stack[state.stack.size-2]
            val x = state.stack[state.stack.size-3]

            // Already evaluated
            if (x is ConstInt) {
                return SkState(
                    code = Ap(Ap(f, x), Ap(g, x)),
                    stack = state.stack.dropLast(3)
                )
            }

            // Start to evaluate x. When it resolves to a value, we will continue executing the S.
            // See also the case for ConstInt
            return SkState(
                code = x,
                // Keep f and g on the stack for later resumption
                stack = state.stack.dropLast(3) + f + g
            )
        }
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

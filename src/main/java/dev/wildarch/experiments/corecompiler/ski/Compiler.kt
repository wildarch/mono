package dev.wildarch.experiments.corecompiler.ski

import dev.wildarch.experiments.corecompiler.syntax.*

sealed class Comb

data class CAp(val func: Comb, val arg: Comb) : Comb()
data class ConstInt(val n: Int) : Comb()
data class FuncRef(val name: String) : Comb()
object I : Comb()
object S : Comb()
object K : Comb()

typealias CombProgram = Map<String, Comb>

private fun compileC(expr: Expr): Comb {
    return when (expr) {
        is Ap -> CAp(compileC(expr.func), compileC(expr.arg))
        is Lam -> {
            var exprC = compileC(expr.body)
            for (param in expr.params.reversed()) {
                exprC = compileA(param, exprC)
            }
            exprC
        }
        is Num -> ConstInt(expr.value)
        is Var -> when (expr.name) {
            "I" -> I
            "K" -> K
            "S" -> S
            else -> FuncRef(expr.name)
        }

        is BinOp -> TODO()
        is Case -> TODO()
        is Constr -> TODO()
        is Let -> TODO()
    }
}

private fun compileA(param: String, expr: Comb): Comb {
    return when (expr) {
        is CAp -> CAp(CAp(S, compileA(param, expr.func)), compileA(param, expr.arg))
        is FuncRef ->
            if (expr.name == param) {
                I
            } else {
                CAp(K, expr)
            }

        else -> CAp(K, expr)
    }
}

fun compile(program: Program): CombProgram {
    return buildMap {
        for (def in program.defs) {
            put(def.name, compileC(Lam(def.params, def.body)))
        }
    }
}

data class SkState(val program: CombProgram, val comb: Comb, val stack: List<Comb>)

fun evaluate(program: CombProgram, maxSteps: Int = 10000): List<SkState> {
    val trace = mutableListOf(SkState(program, program["main"] ?: error("missing main"), emptyList()))
    var steps = 0
    while (!isFinalState(trace.last())) {
        if (steps > maxSteps) {
            error("Did not terminate after $maxSteps")
        }
        trace.add(step(trace.last()))
        steps++;
    }
    return trace
}

private fun isFinalState(state: SkState) = state.comb is ConstInt

private fun step(state: SkState): SkState {
    return when (val comb = state.comb) {
        is CAp -> state.copy(
            comb = comb.func,
            stack = state.stack + comb.arg
        )
        is FuncRef -> state.copy(
            comb = state.program[comb.name] ?: error("missing function ${comb.name}")
        )
        I -> state.copy(
            comb = state.stack.last(),
            stack = state.stack.dropLast(1)
        )
        K -> state.copy(
            comb = state.stack.last(),
            stack = state.stack.dropLast(2)
        )
        S -> {
            val f = state.stack[state.stack.size - 1]
            val g = state.stack[state.stack.size - 2]
            val x = state.stack[state.stack.size - 3]
            state.copy(
                comb = CAp(CAp(f, x), CAp(g, x)),
                stack = state.stack.dropLast(3)
            )
        }
        is ConstInt -> error("Cannot reduce int further")
    }
}

package dev.wildarch.experiments.corecompiler.gmachine

import dev.wildarch.experiments.corecompiler.prelude.preludeDefs
import dev.wildarch.experiments.corecompiler.syntax.*

fun eval(initialState: GmState): List<GmState> {
    val trace = mutableListOf(initialState)
    while (!trace.last().isFinal()) {
        trace.add(doAdmin(step(trace.last())))
    }
    return trace
}

private fun step(state: GmState): GmState {
    val currentInstruction = state.code.first()
    val nextInstructions = state.code.drop(1)
    val nextState = state.copy(code = nextInstructions)
    return dispatch(currentInstruction, nextState)
}

private fun dispatch(instruction: Instruction, nextState: GmState) = when (instruction) {
    is Pushglobal -> pushGlobal(instruction.name, nextState)
    is Pushint -> pushInt(instruction.n, nextState)
    is MkAp -> makeAp(nextState)
    is Push -> push(instruction.n, nextState)
    is Slide -> slide(instruction.n, nextState)
    is Unwind -> unwind(nextState)
}

private fun pushGlobal(name: Name, state: GmState): GmState {
    val addr = state.globals[name] ?: error("Undeclared global $name")
    return state.pushStack(addr)
}

fun pushInt(n: Int, state: GmState): GmState {
    val (newHeap, addr) = heapAlloc(state.heap, NNum(n))
    return state.copy(heap = newHeap).pushStack(addr)
}

private fun makeAp(state: GmState): GmState {
    // Last two elements on stack are func and arg
    val func = state.stack[state.stack.size - 1]
    val arg = state.stack[state.stack.size - 2]
    val (newHeap, addr) = heapAlloc(state.heap, NAp(func, arg))
    val newStack = state.stack.dropLast(2) + addr
    return state.copy(
        heap = newHeap,
        stack = newStack,
    )
}

private fun push(n: Int, state: GmState): GmState {
    val argApAddr = state.stack[state.stack.size - 1 - (n + 1)]
    val argAp = state.heap[argApAddr] as NAp
    val newStack = state.stack + argAp.arg
    return state.copy(
        stack = newStack,
    )
}

private fun slide(n: Int, state: GmState): GmState {
    val top = state.stack.last()
    val newStack = state.stack.dropLast(n + 1) + top
    return state.copy(
        stack = newStack,
    )
}

private fun unwind(state: GmState): GmState {
    val top = state.stack.last()
    return when (val topNode = state.heap[top]!!) {
        // Evaluation is done
        is NNum -> state
        // Continue to unwind from next node in Ap chain
        is NAp -> state.copy(code = listOf(Unwind), stack = state.stack + topNode.func)
        is NGlobal -> {
            assert(state.stack.size > topNode.argc) { "Not enough arguments to supercombinator" }
            return state.copy(code = topNode.code)
        }
    }

}

private fun doAdmin(state: GmState): GmState {
    return state.copy(
        stats = state.stats + 1,
    )
}

private fun heapAlloc(heap: GmHeap, node: Node): Pair<GmHeap, Addr> {
    val addr = 1 + heap.size
    val newHeap = heap.toMutableMap()
    newHeap[addr] = node
    return Pair(newHeap, addr)
}

fun compile(program: Program): GmState {
    val initialCode = listOf(
        Pushglobal("main"),
        Unwind,
    )
    val statInitial = 0
    val scDefs = program.defs + preludeDefs()
    val (heap, globals) = buildInitialHeap(scDefs)
    return GmState(
        code = initialCode,
        stack = emptyList(),
        heap = heap,
        globals = globals,
        stats = statInitial,
    )
}

private fun buildInitialHeap(defs: List<ScDefn>): Pair<GmHeap, GmGlobals> {
    val globals = mutableMapOf<Name, Addr>()
    var heap: GmHeap = emptyMap()
    for (def in defs) {
        val compiled = compileSc(def)
        val (newHeap, addr) = heapAlloc(heap, compiled)
        heap = newHeap
        globals[def.name] = addr
    }
    return Pair(heap, globals)
}

private fun compileSc(def: ScDefn): NGlobal {
    val env = mutableMapOf<Name, Int>()
    def.params.forEachIndexed { index, param -> env[param] = index }
    return NGlobal(def.params.size, compileR(def.body, env, def.params.size))
}

private fun compileR(expr: Expr, env: GmEnv, arity: Int): GmCode {
    return compileC(expr, env) + listOf(Slide(arity + 1), Unwind)
}

private fun compileC(expr: Expr, env: GmEnv): GmCode {
    return when (expr) {
        is Ap -> return compileC(expr.arg, env) + compileC(expr.func, argOffset(env, 1)) + listOf(MkAp)
        is BinOp -> TODO()
        is Case -> TODO()
        is Constr -> TODO()
        is Lam -> TODO()
        is Let -> TODO()
        is Num -> listOf(Pushint(expr.value))
        is Var -> {
            val name = expr.name
            when (val argIndex = env[name]) {
                // Not found in environment, must be a global
                null -> listOf(Pushglobal(name))
                // Local variable
                else -> listOf(Push(argIndex))
            }
        }
    }
}

private fun argOffset(env: GmEnv, offset: Int) = env.mapValues { it.value + offset }

data class GmState(
    val code: GmCode,
    val stack: GmStack,
    val heap: GmHeap,
    val globals: GmGlobals,
    val stats: GmStats,
) {
    fun isFinal() = code.isEmpty()

    fun pushStack(addr: Addr): GmState = this.copy(
        stack = stack + addr
    )
}

typealias GmCode = List<Instruction>
typealias GmStack = List<Addr>
typealias GmHeap = Map<Addr, Node>
typealias GmGlobals = Map<Name, Addr>
typealias GmStats = Int
typealias GmEnv = Map<Name, Int>
typealias Addr = Int
typealias Name = String

sealed class Instruction
object Unwind : Instruction()
data class Pushglobal(val name: Name) : Instruction()
data class Pushint(val n: Int) : Instruction()
data class Push(val n: Int) : Instruction()
object MkAp : Instruction()
data class Slide(val n: Int) : Instruction()

sealed class Node
data class NNum(val n: Int) : Node()
data class NAp(val func: Addr, val arg: Addr) : Node()
data class NGlobal(val argc: Int, val code: GmCode) : Node()
package dev.wildarch.experiments.corecompiler.templateinstantiation

import dev.wildarch.experiments.corecompiler.prelude.preludeDefs
import dev.wildarch.experiments.corecompiler.syntax.*

val extraPreludeDefs: List<ScDefn> = listOf()

fun compile(prog: Program): TiState {
    val scDefs = prog.defs + preludeDefs() + extraPreludeDefs
    val (initialHeap, globals) = buildInitialHeap(scDefs)
    val mainAddr = globals["main"] ?: error("main is not defined")
    val stack = listOf(mainAddr)

    return TiState(stack, TiDump(), initialHeap, globals, stats = 0)
}

private fun buildInitialHeap(scDefs: List<ScDefn>): Pair<Heap, Globals> {
    val heap = mutableMapOf<Addr, Node>()
    val globals = mutableMapOf<Name, Addr>()
    scDefs.forEachIndexed { index, scDefn ->
        heap[index] = NSuperComb(scDefn.name, scDefn.params, scDefn.body)
        globals[scDefn.name] = index
    }
    return Pair(heap, globals)
}

fun eval(state: TiState): List<TiState> {
    val trace = mutableListOf(state)
    while (true) {
        val newState = step(trace.last()) ?: break
        trace.add(updateStats(newState))
    }
    return trace
}

private fun updateStats(state: TiState): TiState {
    return state.copy(
        stats = state.stats + 1
    )
}

// Returns null if the state is final
private fun step(state: TiState): TiState? {
    val node = state.heap[state.stack.last()] ?: error("nothing to step, stack is empty")
    return when (node) {
        is NNum -> if (state.stack.size > 1) {
            error("num on stack, but more addresses follow")
        } else {
            // Nothing left to reduce
            null
        }
        is NAp -> state.copy(
            stack = state.stack + node.func,
        )
        is NSuperComb -> {
            val argsEnd = state.stack.size - 1
            val argsStart = argsEnd - node.args.size
            // TODO: Check if need to reverse
            val argAddrs = state.stack.subList(argsStart, argsEnd).asReversed().map {
                when (val argNode = state.heap[it]) {
                    is NAp -> argNode.arg
                    else -> error("Arg on stack should point to NAp")
                }
            }
            val argMap = node.args.zip(argAddrs).toMap()
            val combGlobals = state.globals + argMap
            val (newHeap, resultAddr) = instantiate(node.body, state.heap, combGlobals)
            val newStack = state.stack.subList(0, argsStart) + resultAddr

            // Add indirection node to skip evaluating the supercombinator a second time
            val newHeapInd = newHeap.toMutableMap()
            // Instead of an Ap for the first argument, next time we will jump straight to the result
            newHeapInd[state.stack[argsStart]] = NInd(resultAddr)

            return state.copy(
                stack = newStack,
                heap = newHeapInd,
            )
        }
        is NInd -> {
            val newStack = state.stack.subList(0, state.stack.size - 1) + node.addr
            return state.copy(
                stack = newStack,
            )
        }
    }
}

private fun instantiate(body: Expr, heap: Heap, env: Globals): Pair<Heap, Addr> {
    return when (body) {
        is Num -> heapAlloc(heap, NNum(body.value))
        is Ap -> {
            val (heap1, addr_fun) = instantiate(body.func, heap, env)
            val (heap2, addr_arg) = instantiate(body.arg, heap1, env)
            return heapAlloc(heap2, NAp(addr_fun, addr_arg))
        }
        is Var -> {
            val addr = env[body.name] ?: error("Undefined name ${body.name}")
            return Pair(heap, addr)
        }
        is Let -> if (body.isRec) {
            val letEnv = env.toMutableMap()
            var nextAddr = heap.size + 1

            // Add all defs to the environment first
            for (def in body.defs) {
                val defSize = heapSize(def.expr)
                if (defSize == 0) {
                    val defVar = def.expr as Var
                    letEnv[def.name] = env[defVar.name] ?: error("Undefined name ${defVar.name}")
                } else {
                    nextAddr += defSize
                    // Point to the element last added to the heap
                    letEnv[def.name] = nextAddr - 1
                }
            }

            var newHeap = heap
            for (def in body.defs) {
                val (h, addr) = instantiate(def.expr, newHeap, letEnv)
                newHeap = h
                letEnv[def.name] = addr
            }
            return instantiate(body.body, newHeap, letEnv)
        } else {
            var newHeap = heap
            val letEnv = env.toMutableMap()
            for (def in body.defs) {
                val (h, addr) = instantiate(def.expr, newHeap, env)
                newHeap = h
                letEnv[def.name] = addr
            }
            return instantiate(body.body, newHeap, letEnv)
        }
        else -> error("Cannot instantiate: $body")
    }
}

private fun heapSize(e: Expr): Int {
    return when (e) {
        is Num -> 1
        is Ap -> heapSize(e.func) + heapSize(e.arg) + 1
        is Var -> 0
        is Let -> e.defs.sumOf { def -> heapSize(def.expr) } + heapSize(e.body)
        else -> error("Unknown heap size: $e")
    }
}

private fun heapAlloc(heap: Heap, node: Node): Pair<Heap, Addr> {
    val addr = 1 + heap.size
    val newHeap = heap.toMutableMap()
    newHeap[addr] = node
    return Pair(newHeap, addr)
}

fun showResults(states: List<TiState>): String {
    // Need to check if this looks okay
    return states.toString()
}

typealias Addr = Int
typealias Name = String
typealias Heap = Map<Addr, Node>
typealias Globals = Map<Name, Addr>

data class TiState(
    val stack: List<Addr>,
    val dump: TiDump,
    val heap: Map<Addr, Node>,
    val globals: Map<Name, Addr>,
    val stats: Int
)

class TiDump

sealed class Node
data class NAp(val func: Addr, val arg: Addr) : Node()
data class NSuperComb(val name: Name, val args: List<Name>, val body: Expr) : Node()
data class NNum(val num: Int) : Node()
data class NInd(val addr: Addr) : Node()
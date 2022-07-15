package dev.wildarch.experiments.corecompiler.templateinstantiation

import dev.wildarch.experiments.corecompiler.prelude.preludeDefs
import dev.wildarch.experiments.corecompiler.syntax.*

val extraPreludeDefs: List<ScDefn> = listOf()

fun compile(prog: Program): TiState {
    val scDefs = prog.defs + preludeDefs() + extraPreludeDefs
    val (initialHeap, globals) = buildInitialHeap(scDefs)
    val mainAddr = globals["main"] ?: error("main is not defined")
    val stack = listOf(mainAddr)

    return TiState(stack, listOf(), initialHeap, globals, stats = 0)
}

private fun buildInitialHeap(scDefs: List<ScDefn>): Pair<Heap, Globals> {
    var heap = mapOf<Addr, Node>()
    val globals = mutableMapOf<Name, Addr>()
    for (def in scDefs) {
        val (newHeap, addr) = heapAlloc(heap, NSuperComb(def.name, def.params, def.body))
        heap = newHeap
        globals[def.name] = addr
    }
    for ((name, prim) in PRIMITIVES) {
        val (newHeap, addr) = heapAlloc(heap, NPrim(name, prim))
        heap = newHeap
        globals[name] = addr
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
    val nodeAddr = state.stack.last()
    val node = state.heap[nodeAddr] ?: error("nothing to step, stack is empty")
    return when (node) {
        is NNum -> if (state.stack.size > 1) {
            error("num on stack, but more addresses follow")
        } else if (state.dump.isNotEmpty()) {
            // Restore previous stack from dump
            return state.copy(
                stack = state.dump.last(),
                dump = state.dump.dropLast(1),
            )
        } else {
            // Nothing left to reduce
            null
        }
        is NData -> if (state.stack.size > 1) {
            error("data on stack, but more addresses follow")
        } else if (state.dump.isNotEmpty()) {
            // Restore previous stack from dump
            return state.copy(
                stack = state.dump.last(),
                dump = state.dump.dropLast(1),
            )
        } else {
            // Nothing left to reduce
            null
        }
        is NAp -> {
            when (val arg = state.heap[node.arg]) {
                // Special case to remove indirection on arg addrs (rule 2.8)
                is NInd -> {
                    val newHeap = state.heap.toMutableMap()
                    newHeap[nodeAddr] = NAp(node.func, arg.addr)
                    return state.copy(
                        heap = newHeap,
                    )
                }
                else -> state.copy(
                    stack = state.stack + node.func,
                )
            }
        }
        is NSuperComb -> {
            val argsEnd = state.stack.size - 1
            val argsStart = argsEnd - node.args.size
            if (argsStart < 0) {
                error("Not enough args")
            }
            val argAddrs = state.stack.subList(argsStart, argsEnd).asReversed().map {
                when (val argNode = state.heap[it]) {
                    is NAp -> argNode.arg
                    else -> error("Arg on stack should point to NAp")
                }
            }
            val argMap = node.args.zip(argAddrs).toMap()
            val combGlobals = state.globals + argMap

            // Instantiate over the old redex root
            val newHeap = instantiateAndUpdate(node.body, state.heap, combGlobals, state.stack[argsStart])
            val newStack = state.stack.subList(0, argsStart + 1)

            return state.copy(
                stack = newStack,
                heap = newHeap,
            )
        }
        is NInd -> {
            val newStack = state.stack.subList(0, state.stack.size - 1) + node.addr
            return state.copy(
                stack = newStack,
            )
        }
        is NPrim -> {
            when (node.prim) {
                is PrimNeg -> {
                    assert(state.stack.size == 2)
                    // The node on the stack before the NPrim must be NAp with the argument
                    val argApAddr = state.stack.first()
                    val argAp = state.heap[argApAddr] as NAp
                    val argNode = state.heap[argAp.arg]
                    when (argNode) {
                        is NNum -> {
                            // Argument has already been evaluated
                            val newHeap = state.heap.toMutableMap()
                            newHeap[argApAddr] = NNum(-argNode.num)
                            val newStack = state.stack.dropLast(1)
                            return state.copy(
                                stack = newStack,
                                heap = newHeap,
                            )
                        }
                        else -> {
                            // Argument needs to be evaluated first
                            val newStack = listOf(argAp.arg)
                            val newDump = state.dump + listOf(listOf(argApAddr))
                            return state.copy(
                                stack = newStack,
                                dump = newDump,
                            )
                        }
                    }
                }
                is PrimIf -> {
                    // Must have three NAp as ancestors
                    assert(state.stack.size == 4)

                    // Condition
                    val argCondApAddr = state.stack[2]
                    val argCondAp = state.heap[argCondApAddr] as NAp
                    val argCondNode = state.heap[argCondAp.arg]
                    if (argCondNode !is NData) {
                        // cond has not been evaluated yet
                        val newStack = listOf(argCondAp.arg)
                        val newDump = state.dump + listOf(listOf(state.stack.first()))
                        return state.copy(
                            stack = newStack,
                            dump = newDump,
                        )
                    }

                    val takenApAddr = when (argCondNode.tag) {
                        1 -> state.stack[0]
                        2 -> state.stack[1]
                        else -> error("Invalid tag for if")
                    }
                    val takenAp = state.heap[takenApAddr] as NAp
                    // This is the address of the expression we want to evaluate
                    val takenAddr = takenAp.arg

                    val newHeap = state.heap.toMutableMap()
                    newHeap[state.stack.first()] = NInd(takenAddr)
                    val newStack = listOf(takenAddr)
                    return state.copy(
                        stack = newStack,
                        heap = newHeap,
                    )
                }
                is PrimConstr -> {
                    val constr = node.prim
                    assert(state.stack.size == constr.arity + 1)
                    // Get the fields
                    val fields = mutableListOf<Addr>()
                    val apAddrs = state.stack.subList(0, state.stack.size - 1)
                    for (apAddr in apAddrs.reversed()) {
                        val ap = state.heap[apAddr] as NAp
                        fields.add(ap.arg)
                    }

                    val newHeap = state.heap.toMutableMap()
                    newHeap[state.stack.first()] = NData(constr.tag, fields)
                    val newStack = listOf(state.stack.first())
                    return state.copy(
                        stack = newStack,
                        heap = newHeap,
                    )
                }
                is PrimCasePair -> {
                    // Must have two NAp as parent and grandparent
                    assert(state.stack.size == 3)

                    val argDataApAddr = state.stack[1]
                    val argDataAp = state.heap[argDataApAddr] as NAp
                    val argData = state.heap[argDataAp.arg]
                    if (argData !is NData) {
                        // Pair has not been evaluated yet
                        val newStack = listOf(argDataAp.arg)
                        val newDump = state.dump + listOf(listOf(state.stack.first()))
                        return state.copy(
                            stack = newStack,
                            dump = newDump,
                        )
                    }
                    assert(argData.tag == 1)
                    assert(argData.fields.size == 2)
                    val lhsAddr = argData.fields[0]
                    val rhsAddr = argData.fields[1]

                    val argFuncApAddr = state.stack[0]
                    val argFuncAp = state.heap[argFuncApAddr] as NAp

                    val newHeap = state.heap.toMutableMap()
                    newHeap[nodeAddr] = NInd(argFuncAp.arg)
                    newHeap[argDataApAddr] = NAp(argFuncAp.arg, lhsAddr)
                    newHeap[argFuncApAddr] = NAp(argDataApAddr, rhsAddr)

                    val newStack = listOf(argFuncApAddr)

                    return state.copy(
                        stack = newStack,
                        heap = newHeap,
                    )
                }
                else -> {
                    // Must have two NAp as parent and grandparent
                    assert(state.stack.size == 3)
                    val argLhsApAddr = state.stack[1]
                    val argLhsAp = state.heap[argLhsApAddr] as NAp
                    val argLhsNode = state.heap[argLhsAp.arg]
                    if (argLhsNode !is NNum) {
                        assert(argLhsNode !is NData)
                        // Left-hand side has not been evaluated yet
                        val newStack = listOf(argLhsAp.arg)
                        val newDump = state.dump + listOf(listOf(state.stack.first()))
                        return state.copy(
                            stack = newStack,
                            dump = newDump,
                        )
                    }
                    val argRhsApAddr = state.stack.first()
                    val argRhsAp = state.heap[argRhsApAddr] as NAp
                    val argRhsNode = state.heap[argRhsAp.arg]
                    if (argRhsNode !is NNum) {
                        assert(argRhsNode !is NData)
                        // Right-hand side has not been evaluated yet
                        val newStack = listOf(argRhsAp.arg)
                        val newDump = state.dump + listOf(listOf(state.stack.first()))
                        return state.copy(
                            stack = newStack,
                            dump = newDump,
                        )
                    }
                    // Both arguments have been evaluated
                    val lhsVal = argLhsNode.num
                    val rhsVal = argRhsNode.num
                    val result = when (node.prim) {
                        is PrimAdd -> NNum(lhsVal + rhsVal)
                        is PrimSub -> NNum(lhsVal - rhsVal)
                        is PrimMul -> NNum(lhsVal * rhsVal)
                        is PrimDiv -> NNum(lhsVal / rhsVal)
                        PrimEq -> toNBool(lhsVal == rhsVal)
                        PrimGt -> toNBool(lhsVal > rhsVal)
                        PrimGte -> toNBool(lhsVal >= rhsVal)
                        PrimLt -> toNBool(lhsVal < rhsVal)
                        PrimLte -> toNBool(lhsVal <= rhsVal)
                        PrimNeq -> toNBool(lhsVal != rhsVal)
                        else -> error("Not a binary primitive")
                    }
                    val newHeap = state.heap.toMutableMap()
                    newHeap[argRhsApAddr] = result
                    val newStack = state.stack.dropLast(2)
                    return state.copy(
                        stack = newStack,
                        heap = newHeap,
                    )
                }
            }
        }
    }
}

private fun instantiate(body: Expr, heap: Heap, env: Globals): Pair<Heap, Addr> {
    val (heap1, node) = instantiateNode(body, heap, env)
    return when (node) {
        is NInd -> Pair(heap1, node.addr)
        else -> heapAlloc(heap1, node)
    }
}

private fun instantiateAndUpdate(body: Expr, heap: Heap, env: Globals, toUpdate: Addr): Heap {
    val (heap1, node) = instantiateNode(body, heap, env)
    val heap2 = heap1.toMutableMap()
    heap2[toUpdate] = node
    return heap2
}

// Instantiates body, returning an updated heap and the node representing the expression.
private fun instantiateNode(body: Expr, heap: Heap, env: Globals): Pair<Heap, Node> {
    return when (body) {
        is Num -> Pair(heap, NNum(body.value))
        is Ap -> {
            val (heap1, addrFun) = instantiate(body.func, heap, env)
            val (heap2, addrArg) = instantiate(body.arg, heap1, env)
            return Pair(heap2, NAp(addrFun, addrArg))
        }
        is Var -> {
            val addr = env[body.name] ?: error("Undefined name ${body.name}")
            return Pair(heap, NInd(addr))
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
            return instantiateNode(body.body, newHeap, letEnv)
        } else {
            var newHeap = heap
            val letEnv = env.toMutableMap()
            for (def in body.defs) {
                val (h, addr) = instantiate(def.expr, newHeap, env)
                newHeap = h
                letEnv[def.name] = addr
            }
            return instantiateNode(body.body, newHeap, letEnv)
        }
        is BinOp -> {
            val prim = when (body.op) {
                Operator.ADD -> NPrim("+", PrimAdd)
                Operator.SUB -> NPrim("-", PrimSub)
                Operator.MUL -> NPrim("*", PrimMul)
                Operator.DIV -> NPrim("/", PrimDiv)
                Operator.EQ -> NPrim("==", PrimEq)
                Operator.NEQ -> NPrim("~=", PrimNeq)
                Operator.GT -> NPrim(">", PrimGt)
                Operator.GTE -> NPrim(">=", PrimGte)
                Operator.LT -> NPrim("<", PrimLt)
                Operator.LTE -> NPrim("<=", PrimLte)
                Operator.AND -> TODO()
                Operator.OR -> TODO()
            }
            val (heap1, addrPrim) = heapAlloc(heap, prim)
            val (heap2, addrArgLhs) = instantiate(body.lhs, heap1, env)
            val (heap3, addrAp1) = heapAlloc(heap2, NAp(addrPrim, addrArgLhs))
            val (heap4, addrArgRhs) = instantiate(body.rhs, heap3, env)
            return Pair(heap4, NAp(addrAp1, addrArgRhs))
        }
        is Constr -> Pair(heap, NPrim("Pack", PrimConstr(body.tag, body.arity)))
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
typealias TiStack = List<Addr>
typealias TiDump = List<TiStack>

data class TiState(
    val stack: TiStack, val dump: TiDump, val heap: Map<Addr, Node>, val globals: Map<Name, Addr>, val stats: Int
)

sealed class Node
data class NAp(val func: Addr, val arg: Addr) : Node()
data class NSuperComb(val name: Name, val args: List<Name>, val body: Expr) : Node()
data class NNum(val num: Int) : Node()
data class NInd(val addr: Addr) : Node()
data class NPrim(val name: String, val prim: Primitive) : Node()
data class NData(val tag: Int, val fields: List<Addr>) : Node()

sealed class Primitive
object PrimNeg : Primitive()
object PrimAdd : Primitive()
object PrimSub : Primitive()
object PrimMul : Primitive()
object PrimDiv : Primitive()
data class PrimConstr(val tag: Int, val arity: Int) : Primitive()
object PrimIf : Primitive()
object PrimGt : Primitive()
object PrimGte : Primitive()
object PrimLt : Primitive()
object PrimLte : Primitive()
object PrimEq : Primitive()
object PrimNeq : Primitive()
object PrimCasePair : Primitive()

val PRIMITIVES = mapOf(
    "negate" to PrimNeg,
    "+" to PrimAdd,
    "-" to PrimSub,
    "*" to PrimMul,
    "/" to PrimDiv,
    "if" to PrimIf,
    ">" to PrimGt,
    ">=" to PrimGte,
    "<" to PrimLt,
    "<=" to PrimLte,
    "==" to PrimEq,
    "~=" to PrimNeq,
    "casePair" to PrimCasePair,
)

val NFALSE = NData(1, listOf())
val NTRUE = NData(2, listOf())

private fun toNBool(b: Boolean): NData = if (b) NTRUE else NFALSE

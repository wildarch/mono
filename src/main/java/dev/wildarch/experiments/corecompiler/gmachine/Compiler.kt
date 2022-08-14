package dev.wildarch.experiments.corecompiler.gmachine

import dev.wildarch.experiments.corecompiler.prelude.preludeDefs
import dev.wildarch.experiments.corecompiler.syntax.*

fun evaluate(initialState: GmState, maxSteps: Int = 10000): List<GmState> {
    val trace = mutableListOf(initialState)
    var steps = 0
    while (!trace.last().isFinal()) {
        if (steps > maxSteps) {
            error("Did not terminate after $maxSteps")
        }
        trace.add(doAdmin(step(trace.last())))
        steps++;
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
    is Update -> update(instruction.n, nextState)
    is Pop -> pop(instruction.n, nextState)
    is Alloc -> alloc(instruction.n, nextState)
    is Eval -> eval(nextState)
    Add -> primBinOp(Add, nextState)
    is Cond -> TODO()
    Div -> TODO()
    Eq -> TODO()
    Ge -> TODO()
    Gt -> TODO()
    Le -> TODO()
    Lt -> TODO()
    Mul -> TODO()
    Ne -> TODO()
    Neg -> primNeg(nextState)
    Sub -> TODO()
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
    val argAddr = state.stack[state.stack.size - 1 - n]
    val newStack = state.stack + argAddr
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
        is NNum -> when (val dumpEntry = state.dump.lastOrNull()) {
            // Done
            null -> return state
            else -> {
                assert(state.stack.size == 1)
                val newCode = dumpEntry.first
                val newStack = dumpEntry.second.toMutableList()
                newStack.add(top)
                return state.copy(
                    code = newCode,
                    stack = newStack,
                    dump = state.dump.dropLast(1),
                )
            }
        }
        // Continue to unwind from next node in Ap chain
        is NAp -> state.copy(code = listOf(Unwind), stack = state.stack + topNode.func)
        is NGlobal -> {
            assert(state.stack.size > topNode.argc) { "Not enough arguments to supercombinator" }
            // Rearrange the stack
            val argAddrs = state.stack
                .dropLast(1)                        // Drop the function
                .takeLast(topNode.argc)                // Take the addrs to the Ap nodes containing the arguments
                .map { (state.heap[it] as NAp).arg }   // Convert the Ap addrs into argument addrs
            val newStack = state.stack.dropLast(topNode.argc) + argAddrs
            return state.copy(code = topNode.code, stack = newStack)
        }
        is NInd -> state.copy(code = listOf(Unwind), stack = state.stack.dropLast(1) + topNode.addr)
    }
}

private fun update(n: Int, state: GmState): GmState {
    val resultAddr = state.stack.last()
    val addrToUpdate = state.stack[state.stack.size - 1 - (n + 1)]
    val newHeap = state.heap.toMutableMap()
    newHeap[addrToUpdate] = NInd(resultAddr)
    // Drop resultAddr
    val newStack = state.stack.dropLast(1)
    return state.copy(stack = newStack, heap = newHeap)
}

private fun pop(n: Int, state: GmState): GmState {
    return state.copy(stack = state.stack.dropLast(n))
}

private fun alloc(n: Int, state: GmState): GmState {
    var newHeap = state.heap
    val newStack = state.stack.toMutableList()
    for (i in 0 until n) {
        val (heap, addr) = heapAlloc(newHeap, NInd(-1))
        newHeap = heap
        newStack.add(addr)
    }
    return state.copy(
        stack = newStack,
        heap = newHeap,
    )
}

private fun eval(state: GmState): GmState {
    val newCode = listOf(Unwind)
    val newStack = listOf(state.stack.last())
    val newDump = state.dump + Pair(state.code, state.stack.dropLast(1))
    return state.copy(
        code = newCode,
        stack = newStack,
        dump = newDump
    )
}

private fun primNeg(state: GmState): GmState {
    val num = state.heap[state.stack.last()] as NNum
    val (newHeap, addr) = heapAlloc(state.heap, NNum(-num.n))
    val newStack = state.stack.dropLast(1) + addr
    return state.copy(
        stack = newStack,
        heap = newHeap,
    )
}

private fun primBinOp(op: PrimBinary, state: GmState): GmState {
    val lhs = (state.heap[state.stack[state.stack.size - 1]] as NNum).n
    val rhs = (state.heap[state.stack[state.stack.size - 2]] as NNum).n
    val res = when (op) {
        Add -> lhs + rhs
        Div -> TODO()
        Eq -> TODO()
        Ge -> TODO()
        Gt -> TODO()
        Le -> TODO()
        Lt -> TODO()
        Mul -> TODO()
        Ne -> TODO()
        Sub -> TODO()
    }
    val (newHeap, addr) = heapAlloc(state.heap, NNum(res))
    val newStack = state.stack.dropLast(2) + addr
    return state.copy(
        stack = newStack,
        heap = newHeap,
    )
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
        dump = emptyList(),
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
    for (prim in COMPILED_PRIMITIVES) {
        val (name, compiled) = prim
        val (newHeap, addr) = heapAlloc(heap, compiled)
        heap = newHeap
        globals[name] = addr
    }
    return Pair(heap, globals)
}

private fun compileSc(def: ScDefn): NGlobal {
    val env = mutableMapOf<Name, Int>()
    def.params.forEachIndexed { index, param -> env[param] = index }
    return NGlobal(def.params.size, compileR(def.body, env, def.params.size))
}

private fun compileR(expr: Expr, env: GmEnv, arity: Int): GmCode {
    return compileC(expr, env) + listOf(Update(arity), Pop(arity), Unwind)
}

private fun compileC(expr: Expr, env: GmEnv): GmCode {
    return when (expr) {
        is Ap -> return compileC(expr.arg, env) + compileC(expr.func, argOffset(env, 1)) + listOf(MkAp)
        is BinOp -> return compileC(expr.rhs, env) + compileC(expr.lhs, argOffset(env, 1)) + listOf(Pushglobal(primitiveFor(expr.op)), MkAp, MkAp)
        is Case -> TODO()
        is Constr -> TODO()
        is Lam -> TODO()
        is Let -> if (expr.isRec) {
            val letBinds = buildMap {
                expr.defs.forEachIndexed { index, def ->
                    put(def.name, expr.defs.size - 1 - index)
                }
            }
            val recEnv = argOffset(env, expr.defs.size) + letBinds
            return listOf(Alloc(expr.defs.size)) + expr.defs.flatMapIndexed { index, def ->
                compileC(def.expr, recEnv) + Update(expr.defs.size - 1 - index)
            } + compileC(expr.body, recEnv) + listOf(Slide(expr.defs.size))
        } else {
            val letBinds = buildMap {
                expr.defs.forEachIndexed { index, def ->
                    put(def.name, expr.defs.size - 1 - index)
                }
            }
            return expr.defs.flatMapIndexed { index, def ->
                compileC(def.expr, argOffset(env, index))
            } + compileC(expr.body, argOffset(env, expr.defs.size) + letBinds) + listOf(Slide(expr.defs.size))
        }
        is Num -> listOf(Pushint(expr.value))
        is Var -> {
            val name = expr.name
            // Is it a local variable?
            val argIndex = env[name]
            if (argIndex != null) {
                return listOf(Push(argIndex))
            }
            // Assume it is a global then
            return listOf(Pushglobal(name))
        }
    }
}

private val COMPILED_PRIMITIVES = mapOf(
    "negate" to NGlobal(1, listOf(Push(0), Eval, Neg, Update(1), Pop(1), Unwind)),
    "+" to NGlobal(2, listOf(Push(1), Eval, Push(1), Eval, Add, Update(2), Pop(2), Unwind))
)

private fun primitiveFor(op: Operator) = when(op) {
    Operator.ADD -> "+"
    Operator.SUB -> TODO()
    Operator.MUL -> TODO()
    Operator.DIV -> TODO()
    Operator.EQ -> TODO()
    Operator.NEQ -> TODO()
    Operator.GT -> TODO()
    Operator.GTE -> TODO()
    Operator.LT -> TODO()
    Operator.LTE -> TODO()
    Operator.AND -> TODO()
    Operator.OR -> TODO()
}

private fun argOffset(env: GmEnv, offset: Int) = env.mapValues { it.value + offset }

data class GmState(
    val code: GmCode,
    val stack: GmStack,
    val dump: GmDump,
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
typealias GmDump = List<Pair<GmCode, GmStack>>
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
data class Update(val n: Int) : Instruction()
data class Pop(val n: Int) : Instruction()
data class Alloc(val n: Int) : Instruction()
object Eval : Instruction()
object Neg : Instruction()
data class Cond(val trueBranch: GmCode, val falseBranch: GmCode) : Instruction()
sealed class PrimBinary : Instruction()
object Add : PrimBinary()
object Sub : PrimBinary()
object Mul : PrimBinary()
object Div : PrimBinary()
object Eq : PrimBinary()
object Ne : PrimBinary()
object Lt : PrimBinary()
object Le : PrimBinary()
object Gt : PrimBinary()
object Ge : PrimBinary()

sealed class Node
data class NNum(val n: Int) : Node()
data class NAp(val func: Addr, val arg: Addr) : Node()
data class NGlobal(val argc: Int, val code: GmCode) : Node()
data class NInd(val addr: Addr) : Node()
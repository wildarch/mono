package dev.wildarch.experiments.corecompiler.tim

import dev.wildarch.experiments.corecompiler.prelude.preludeDefs
import dev.wildarch.experiments.corecompiler.syntax.*

fun compile(program: Program): TimState {
    val initialInstructions = listOf(Enter(Label("main")))
    val scDefs = preludeDefs() + program.defs
    val initialEnv = buildMap {
        for (def in scDefs) {
            put(def.name, Label(def.name))
        }
        for (key in COMPILED_PRIMITIVES.keys) {
            put(key, Label(key))
        }
    }
    val compiledScDefs = buildMap {
        for (def in scDefs) {
            put(def.name, compileSc(initialEnv, def))
        }
    }
    // TODO: Compiled primitives
    val compiledCode = COMPILED_PRIMITIVES + compiledScDefs

    return TimState(
        instructions = initialInstructions,
        framePointer = FrameNull,
        stack = listOf(Closure(emptyList(), FrameNull)),
        valueStack = emptyList(),
        heap = emptyMap(),
        codeStore = compiledCode,
    )
}

private fun compileSc(env: TimCompilerEnv, def: ScDefn): List<Instruction> {
    val newEnv = buildMap {
        def.params.forEachIndexed { index, param ->
            put(param, Arg(index))
        }
        putAll(env)
    }
    return buildList {
        if (def.params.isNotEmpty()) {
            // Only create a new frame if there are actual parameters to store in the frame
            add(Take(def.params.size))
        }
        addAll(compileR(def.body, newEnv))
    }
}

private fun compileR(expr: Expr, env: TimCompilerEnv): List<Instruction> {
    return when (expr) {
        is Ap -> listOf(Push(compileA(expr.arg, env))) + compileR(expr.func, env)
        is Var -> listOf(Enter(compileA(expr, env)))
        is Num -> listOf(Enter(compileA(expr, env)))
        is BinOp -> listOf(
            Push(compileA(expr.rhs, env)), Push(compileA(expr.lhs, env)), Enter(Label(labelFor(expr.op)))
        )
        is Case -> TODO()
        is Constr -> TODO()
        is Lam -> TODO()
        is Let -> TODO()
    }
}

private fun compileA(expr: Expr, env: TimCompilerEnv): TimAMode {
    return when (expr) {
        is Var -> env[expr.name] ?: error("Unknown variable ${expr.name}")
        is Num -> IntConst(expr.value)
        else -> Code(compileR(expr, env))
    }
}

private fun labelFor(op: Operator) = when (op) {
    Operator.ADD -> "+"
    Operator.SUB -> "-"
    Operator.MUL -> "*"
    Operator.DIV -> "/"
    Operator.EQ -> "=="
    Operator.NEQ -> "~="
    Operator.GT -> ">"
    Operator.GTE -> ">="
    Operator.LT -> "<"
    Operator.LTE -> "<="
    Operator.AND -> "&"
    Operator.OR -> "|"
}

private val COMPILED_PRIMITIVES = mapOf(
    "+" to primitiveForBinOp(OpKind.ADD),
    "-" to primitiveForBinOp(OpKind.SUB),
    "*" to primitiveForBinOp(OpKind.MULT),
    "/" to primitiveForBinOp(OpKind.DIV),
    "==" to primitiveForBinOp(OpKind.EQ),
    "~=" to primitiveForBinOp(OpKind.NOT_EQ),
    ">" to primitiveForBinOp(OpKind.GR),
    ">=" to primitiveForBinOp(OpKind.GR_EQ),
    "<" to primitiveForBinOp(OpKind.LT),
    "<=" to primitiveForBinOp(OpKind.LT_EQ),
    "negate" to listOf(
        Take(1), Push(
            Code(
                listOf(
                    Op(OpKind.NEG), Return
                )
            )
        ), Enter(Arg(0))
    )
)

private fun primitiveForBinOp(opKind: OpKind): List<Instruction> {
    return listOf(
        Take(2),
        Push(
            Code(
                listOf(
                    Push(
                        Code(
                            listOf(
                                Op(opKind),
                                Return,
                            )
                        )
                    ),
                    Enter(Arg(0)),
                )
            )
        ),
        Enter(Arg(1)),
    )
}

fun evaluate(initialState: TimState, maxSteps: Int = 10000): List<TimState> {
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

private fun step(state: TimState): TimState {
    return when (val inst = state.instructions.first()) {
        is Take -> {
            assert(state.stack.size >= inst.n)
            val newStack = state.stack.dropLast(inst.n)
            val (newHeap, newFramePtr) = heapAlloc(state.heap, state.stack.takeLast(inst.n).reversed())
            state.copy(
                instructions = state.instructions.drop(1),
                framePointer = FrameAddr(newFramePtr),
                stack = newStack,
                heap = newHeap,
            )
        }
        is Enter -> {
            val closure = argToClosure(inst.arg, state.framePointer, state.heap, state.codeStore)
            state.copy(
                instructions = closure.code,
                framePointer = closure.framePtr,
            )
        }
        is Push -> {
            val closure = argToClosure(inst.arg, state.framePointer, state.heap, state.codeStore)
            val newInstructions = state.instructions.drop(1)
            val newStack = state.stack + closure
            state.copy(
                instructions = newInstructions,
                stack = newStack,
            )
        }
        is Op -> {
            // Handle unary op first
            if (inst.opKind == OpKind.NEG) {
                val result = -state.valueStack.last()
                val newValueStack = state.valueStack.dropLast(1) + result
                val newInstructions = state.instructions.drop(1)
                return state.copy(
                    instructions = newInstructions,
                    valueStack = newValueStack,
                )
            }
            val lhs = state.valueStack[state.valueStack.size - 1]
            val rhs = state.valueStack[state.valueStack.size - 2]
            val result = when (inst.opKind) {
                OpKind.ADD -> lhs + rhs
                OpKind.SUB -> lhs - rhs
                OpKind.MULT -> lhs * rhs
                OpKind.DIV -> lhs / rhs
                OpKind.GR -> lhs > rhs
                OpKind.GR_EQ -> lhs >= rhs
                OpKind.LT -> lhs < rhs
                OpKind.LT_EQ -> lhs <= rhs
                OpKind.EQ -> lhs == rhs
                OpKind.NOT_EQ -> lhs != rhs
                OpKind.NEG -> error("Special case neg not handled")
            }
            val resultInt = when (result) {
                is Int -> result
                true -> 1
                false -> 0
                else -> error("Invalid result type ${result.javaClass}")
            }
            val newValueStack = state.valueStack.dropLast(2) + resultInt
            val newInstructions = state.instructions.drop(1)
            return state.copy(
                instructions = newInstructions,
                valueStack = newValueStack,
            )
        }
        is PushV -> {
            val value = when (val arg = inst.arg) {
                ValueAMode.FramePtr -> (state.framePointer as FrameInt).value
                is ValueAMode.IntConst -> arg.n
            }
            val newValueStack = state.valueStack + value
            val newInstructions = state.instructions.drop(1)
            return state.copy(
                instructions = newInstructions,
                valueStack = newValueStack,
            )
        }
        Return -> {
            val topClosure = state.stack.last()
            val newStack = state.stack.dropLast(1)
            return state.copy(
                instructions = topClosure.code,
                framePointer = topClosure.framePtr,
                stack = newStack,
            )
        }
    }
}

val INT_CODE = listOf(
    PushV(ValueAMode.FramePtr),
    Return,
)

private fun argToClosure(arg: TimAMode, framePointer: FramePtr, heap: TimHeap, codeStore: CodeStore): Closure {
    return when (arg) {
        is Arg -> {
            val frame = heap[(framePointer as FrameAddr).a]!!
            frame[arg.n]
        }
        is Code -> Closure(arg.code, framePointer)
        is IntConst -> Closure(INT_CODE, FrameInt(arg.value))
        is Label -> Closure(codeStore[arg.l] ?: error("Unknown label ${arg.l}"), framePointer)
    }
}

private fun heapAlloc(heap: TimHeap, frame: Frame): Pair<TimHeap, Addr> {
    val addr = 1 + heap.size
    val newHeap = heap.toMutableMap()
    newHeap[addr] = frame
    return Pair(newHeap, addr)
}

sealed class Instruction
data class Take(val n: Int) : Instruction()
data class Enter(val arg: TimAMode) : Instruction()
data class Push(val arg: TimAMode) : Instruction()
data class PushV(val arg: ValueAMode) : Instruction()
object Return : Instruction()
data class Op(val opKind: OpKind) : Instruction()

enum class OpKind {
    ADD, SUB, MULT, DIV, NEG, GR, GR_EQ, LT, LT_EQ, EQ, NOT_EQ,
}

sealed class TimAMode
data class Arg(val n: Int) : TimAMode()
data class Label(val l: String) : TimAMode()
data class Code(val code: List<Instruction>) : TimAMode()
data class IntConst(val value: Int) : TimAMode()

sealed class ValueAMode {
    object FramePtr : ValueAMode()
    data class IntConst(val n: Int) : ValueAMode()
}

data class TimState(
    val instructions: List<Instruction>,
    val framePointer: FramePtr,
    val stack: TimStack,
    val valueStack: TimValueStack,
    val heap: TimHeap,
    val codeStore: CodeStore
) {
    fun isFinal() = instructions.isEmpty()
}

sealed class FramePtr
data class FrameAddr(val a: Addr) : FramePtr()
data class FrameInt(val value: Int) : FramePtr()
object FrameNull : FramePtr()

data class Closure(val code: List<Instruction>, val framePtr: FramePtr)

typealias Addr = Int
typealias Name = String
typealias TimStack = List<Closure>
typealias TimValueStack = List<Int>
typealias Frame = List<Closure>
typealias TimHeap = Map<Addr, Frame>
typealias CodeStore = Map<Name, List<Instruction>>
typealias TimCompilerEnv = Map<Name, TimAMode>
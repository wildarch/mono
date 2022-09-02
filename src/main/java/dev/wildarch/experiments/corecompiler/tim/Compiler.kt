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
    }
    val compiledScDefs = buildMap {
        for (def in scDefs) {
            put(def.name, compileSc(initialEnv, def))
        }
    }
    // TODO: Compiled primitives
    val compiledCode = compiledScDefs

    return TimState(
        instructions = initialInstructions,
        framePointer = FrameNull,
        stack = emptyList(),
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
        if(def.params.isNotEmpty()) {
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
        else -> TODO()
    }
}

private fun compileA(expr: Expr, env: TimCompilerEnv): TimAMode {
    return when (expr) {
        is Var -> env[expr.name] ?: error("Unknown variable ${expr.name}")
        is Num -> IntConst(expr.value)
        else -> Code(compileR(expr, env))
    }
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
    }
}

private fun argToClosure(arg: TimAMode, framePointer: FramePtr, heap: TimHeap, codeStore: CodeStore): Closure {
    return when (arg) {
        is Arg -> {
            val frame = heap[(framePointer as FrameAddr).a]!!
            frame[arg.n]
        }
        is Code -> Closure(arg.code, framePointer)
        is IntConst -> Closure(emptyList(), FrameInt(arg.value))
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

sealed class TimAMode
data class Arg(val n: Int) : TimAMode()
data class Label(val l: String) : TimAMode()
data class Code(val code: List<Instruction>) : TimAMode()
data class IntConst(val value: Int) : TimAMode()

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
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
        heap = emptyMap(),
        codeStore = compiledCode,
    )
}

private fun compileSc(env: TimCompilerEnv, def: ScDefn): List<Instruction> {
    val newEnv = buildMap {
        def.params.forEachIndexed { index, param ->
            put(param, Arg(index+1))
        }
        putAll(env)
    }
    return buildList {
        add(Take(def.params.size))
        addAll(compileR(def.body, newEnv))
    }
}

private fun compileR(expr: Expr, env: TimCompilerEnv): List<Instruction> {
    return when(expr) {
        is Ap -> listOf(Push(compileA(expr.arg, env))) + compileR(expr.func, env)
        is Var -> listOf(Enter(compileA(expr, env)))
        is Num -> listOf(Enter(compileA(expr, env)))
        else -> TODO()
    }
}

private fun compileA(expr: Expr, env: TimCompilerEnv): TimAMode {
    return when(expr) {
        is Var -> env[expr.name] ?: error("Unknown variable ${expr.name}")
        is Num -> IntConst(expr.value)
        else -> Code(compileR(expr, env))
    }
}

fun evaluate(initialState: TimState, maxSteps: Int = 10000): List<TimState> {
    TODO()
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
    val heap: TimHeap,
    val codeStore: CodeStore
)

sealed class FramePtr
data class FrameAddr(val a: Addr) : FramePtr()
data class FrameInt(val value: Int) : FramePtr()
object FrameNull : FramePtr()

data class Closure(val code: List<Instruction>, val framePtr: FramePtr)

typealias Addr = Int
typealias Name = String
typealias TimStack = List<Closure>
typealias Frame = List<Closure>
typealias TimHeap = Map<Addr, Frame>
typealias CodeStore = Map<Name, List<Instruction>>
typealias TimCompilerEnv = Map<Name, TimAMode>
package dev.wildarch.experiments.corecompiler.gmachine

data class GmState(
    val code: GmCode,
    val stack: GmStack,
    val heap: GmHeap,
    val globals: GmGlobals,
    val stats: GmStats,
)

typealias GmCode = List<Instruction>
typealias GmStack = List<Addr>
typealias GmHeap = Map<Addr, Node>
typealias GmGlobals = Map<Name, Addr>
typealias GmStats = Int
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
data class NNum(val n: Int) : Instruction()
data class NAp(val func: Addr, val arg: Addr) : Instruction()
data class NGlobal(val argc: Int, val code: GmCode) : Instruction()
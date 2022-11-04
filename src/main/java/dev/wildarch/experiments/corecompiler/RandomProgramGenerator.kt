package dev.wildarch.experiments.corecompiler

import dev.wildarch.experiments.corecompiler.syntax.BinOp
import dev.wildarch.experiments.corecompiler.syntax.Num
import kotlin.random.Random

// 1. Generate a bunch of random binops with constants.
//    Evaluate their results and store it with the expression
//    Keep a separate list of all expressions, with parent expr
// 2. Apply abstractions Let,Lam

fun randomNum(rand: Random): Num = Num(rand.nextInt(-10_000, 10_000))

fun randomBinOp(rand: Random, cost: Int): BinOp {
    when (cost) {
        0 -> error("Cost 0")
        1, 2 -> randomNum(rand)
        x -> TODO("Binop takes 1, split remainder over argument.")
    }
    TODO()
}
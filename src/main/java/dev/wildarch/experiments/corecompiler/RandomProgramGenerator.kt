package dev.wildarch.experiments.corecompiler

import dev.wildarch.experiments.corecompiler.syntax.BinOp
import dev.wildarch.experiments.corecompiler.syntax.Expr
import dev.wildarch.experiments.corecompiler.syntax.Num
import dev.wildarch.experiments.corecompiler.syntax.Operator
import kotlin.random.Random

// 1. Generate a bunch of random binops with constants.
//    Evaluate their results and store it with the expression
//    Keep a separate list of all expressions, with parent expr
// 2. Apply abstractions Let,Lam

fun randomNum(rand: Random, ensureNonZero: Boolean): Num {
    while (true) {
        val n = rand.nextInt(-10_000, 10_000)
        if (n != 0) {
            return Num(n)
        }
    }
}

fun randomBinOp(rand: Random, cost: Int, ensureNonZero: Boolean = false): Expr {
    return when (cost) {
        0 -> error("Cost 0")
        1, 2 -> randomNum(rand, ensureNonZero)
        else -> {
            while (true) {
                // Cost of 1 for the node itself
                val (leftCost, rightCost) = splitCost(rand, cost - 1)
                val allowedOps = listOf(
                    Operator.ADD,
                    Operator.SUB,
                    Operator.MUL,
                    Operator.DIV,
                )
                val op = allowedOps[rand.nextInt(allowedOps.size)]
                val e = BinOp(randomBinOp(rand, leftCost), op, randomBinOp(rand, rightCost))
                if (evalBinOp(e) != 0) {
                    return e
                }
            }
            error("unreachable")
        }
    }
}

private fun evalBinOp(e: Expr): Int {
    return when (e) {
        is Num -> e.value
        is BinOp -> {
            val l = evalBinOp(e.lhs)
            val r = evalBinOp(e.rhs)
            when (e.op) {
                Operator.ADD -> l + r
                Operator.SUB -> l - r
                Operator.MUL -> l * r
                Operator.DIV -> l / r
                else -> error("Not supported: $e")
            }
        }

        else -> error("Not supported: $e")
    }
}

fun main() {
    for (i in 0..100) {
        val e = randomBinOp(Random.Default, 10)
        val result = evalBinOp(e)
        println("Random binop: $e. Cost: $result")
    }
}

private fun splitCost(rand: Random, cost: Int): Pair<Int, Int> {
    val a = rand.nextInt(1, cost)
    val b = cost - a
    assert(a > 0)
    assert(b > 0)
    return Pair(a, b)
}
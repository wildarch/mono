package dev.wildarch.experiments.corecompiler.templateinstantiation

import com.google.common.truth.Truth.assertThat
import dev.wildarch.experiments.corecompiler.parser.parse
import org.junit.Test

class CompilerTest {
    @Test
    fun compileTwice() {
        assertEvaluatesTo("main = twice I 2", 2)
    }

    @Test
    fun compileId() {
        assertEvaluatesTo("main = I 42", 42)
    }

    @Test
    fun compileTrivial() {
        assertEvaluatesTo("main = S K K 3", 3)
    }

    private fun assertEvaluatesTo(program: String, num: Int) {
        val parsed = parse(program)
        val compiled = compile(parsed)
        val trace = eval(compiled)

        showResults(trace)

        val finalState = trace.last()
        assertThat(finalState.stack).hasSize(1)
        val finalAddr = finalState.stack[0]
        val finalNode = finalState.heap[finalAddr] ?: error("Invalid final addr")
        val finalNum = finalNode as NNum
        assertThat(finalNum.num).isEqualTo(num)
    }
}
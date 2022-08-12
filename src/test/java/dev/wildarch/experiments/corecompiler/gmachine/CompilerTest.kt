package dev.wildarch.experiments.corecompiler.gmachine

import com.google.common.truth.Truth
import dev.wildarch.experiments.corecompiler.parser.parse
import org.junit.Test

class CompilerTest {
    @Test
    fun compileId() {
        assertEvaluatesTo("main = I 42", 42)
    }

    @Test
    fun compileTrivial() {
        assertEvaluatesTo("main = S K K 3", 3)
    }

    @Test
    fun compileTwice() {
        assertEvaluatesTo("main = twice I 2", 2)
    }

    private fun assertEvaluatesTo(program: String, num: Int): List<GmState> {
        val parsed = parse(program)
        val compiled = compile(parsed)
        val trace = eval(compiled)

        val finalState = trace.last()
        Truth.assertThat(finalState.stack).hasSize(1)
        val finalAddr = finalState.stack[0]
        val finalNode = finalState.heap[finalAddr] ?: error("Invalid final addr")
        val finalNum = finalNode as NNum
        Truth.assertThat(finalNum.n).isEqualTo(num)
        return trace
    }
}
package dev.wildarch.experiments.corecompiler.tim

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
    fun exampleBasicUltra() {
        assertEvaluatesTo("main = I 3", 3)
        assertEvaluatesTo(
            """
            id = S K K ;
            main = id 3
        """.trimIndent(), 3
        )
        assertEvaluatesTo(
            """
            id = S K K ;
            main = twice twice twice id 3
        """.trimIndent(), 3
        )
    }


    private fun assertEvaluatesTo(program: String, num: Int): List<TimState> {
        val parsed = parse(program)
        val compiled = compile(parsed)
        val trace = evaluate(compiled)

        val finalState = trace.last()
        Truth.assertThat(finalState.framePointer).isEqualTo(FrameInt(num))
        return trace
    }
}
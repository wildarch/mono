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
    fun compileTrivial2() {
        assertEvaluatesTo(
            """
            id1 = S K K ;
            id2 = id1 id1 ;
            main = id1 4
        """.trimIndent(), 4
        )
    }

    @Test
    fun compileNegate() {
        assertEvaluatesTo(
            """
            main = negate 3
        """.trimIndent(), -3
        )
    }

    @Test
    fun compileNegateTwice() {
        assertEvaluatesTo(
            """
            main = twice negate 3
        """.trimIndent(), 3
        )
    }

    @Test
    fun compileAdd() {
        assertEvaluatesTo(
            """
            main = 1 + 2
        """.trimIndent(), 3
        )
    }

    @Test
    fun compileArithmetic() {
        assertEvaluatesTo(
            """
            main = 1 * 2 / 3 - 4
        """.trimIndent(), -4
        )
    }

    @Test
    fun compileNegateIndirection() {
        assertEvaluatesTo(
            """
            main = negate (I 3)
        """.trimIndent(), -3
        )
    }

    @Test
    fun compileSubTwice() {
        assertEvaluatesTo(
            """
             main = (3 - 2) - 1
        """.trimIndent(), 0
        )
    }

    @Test
    fun compileMulIndirect() {
        assertEvaluatesTo(
            """
            main = (I 6) * (I 8)
        """.trimIndent(), 48
        )
    }

    @Test
    fun compileFourPlusFour() {
        assertEvaluatesTo(
            """
                four = 2 * 2 ;
                main = four + four
            """.trimIndent(), 8
        )
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
        Truth.assertThat(finalState.valueStack.last()).isEqualTo(num)
        return trace
    }
}
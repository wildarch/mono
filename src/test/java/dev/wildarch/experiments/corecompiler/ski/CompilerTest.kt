package dev.wildarch.experiments.corecompiler.ski

import com.google.common.truth.Truth
import com.google.common.truth.Truth.assertThat
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
        assertEvaluatesTo("""
            twice x = x x;
            main = twice I 2
        """.trimIndent(), 2)
    }

    @Test
    fun compileLet() {
        assertEvaluatesTo(
            """
             main = 
               let 
                 i = I 
               in 
                 i 42
        """.trimIndent(), 42
        )
    }

    // TODO: letrec

    /*
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
     */

    @Test
    fun exampleBasicInteresting() {
        assertEvaluatesTo(
            """
            cons a b cc cn = cc a b ;
            nil      cc cn = cn ;
            hd list = list K abort ;
            tl list = list K1 abort ;
            Y f = (\x . f (x x)) (\x . f (x x)) ;
            infinite x = Y (cons x) ;
            abort = Y I ;
            
            main = hd (tl (infinite 4))
        """.trimIndent(), 4
        )
    }

}

private fun assertEvaluatesTo(program: String, num: Int): List<SkState> {
    val parsed = parse(program)
    val compiled = compile(parsed)
    println("Compiled: ${compiled}")
    val trace = evaluate(compiled)

    val finalState = trace.last()
    val finalValue = finalState.code
    assertThat(finalValue).isEqualTo(ConstInt(num))
    return trace
}

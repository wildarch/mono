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

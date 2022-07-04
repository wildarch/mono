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

    @Test
    fun compileLetRec() {
        assertEvaluatesTo(
            """
            pair x y f = f x y ;
            fst p = p K ;
            snd p = p K1 ;
            f x y =
              letrec
                a = pair x b ;
                b = pair y a
              in
                fst (snd (snd (snd a))) ;
            main = f 3 4
        """.trimIndent(), 4
        )
    }

    @Test
    fun compileUpdates() {
        val trace = assertEvaluatesTo(
            """
                id x = x ;
                main = twice twice twice id 3
            """.trimIndent(), 3
        )
        val steps = trace.last().stats
        assertThat(steps).isEqualTo(139)
    }

    private fun assertEvaluatesTo(program: String, num: Int): List<TiState> {
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
        return trace
    }
}
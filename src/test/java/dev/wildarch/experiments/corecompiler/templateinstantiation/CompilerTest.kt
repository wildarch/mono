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

    @Test
    fun exampleBasicUpdating() {
        assertEvaluatesTo("main = twice (I I I) 3", 3)
    }

    @Test
    fun exampleBasicInteresting() {
        assertEvaluatesTo(
            """
            cons a b cc cn = cc a b ;
            nil      cc cn = cn ;
            hd list = list K abort ;
            tl list = list K1 abort ;
            abort = abort ;
            
            infinite x = cons x (infinite x) ;
            
            main = hd (tl (infinite 4))
        """.trimIndent(), 4
        )
    }

    @Test
    fun exampleLetRec() {
        assertEvaluatesTo(
            """
            main = let id1 = I I I
                   in id1 id1 3
        """.trimIndent(), 3
        )

        assertEvaluatesTo(
            """
            oct g x = let h = twice g
                      in let k = twice h
                      in k (k x) ;
            main = oct I 4
        """.trimIndent(), 4
        )

        assertEvaluatesTo(
            """
            cons a b cc cn = cc a b ;
            nil      cc cn = cn ;
            hd list = list K abort ;
            tl list = list K1 abort ;
            abort = abort ;
            
            infinite x = letrec xs = cons x xs
                       in xs ;
            
            main = hd (tl (tl (infinite 4)))
        """.trimIndent(), 4
        )
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
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
             main = 
               letrec 
                 i = I 
               in 
                 i 42
        """.trimIndent(), 42
        )

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
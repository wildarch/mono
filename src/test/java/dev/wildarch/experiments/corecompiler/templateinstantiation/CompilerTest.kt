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
    fun compileNegateBasic() {
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
    fun compileNegateIndirection() {
        assertEvaluatesTo(
            """
            main = negate (I 3)
        """.trimIndent(), -3
        )
    }

    @Test
    fun compileAddBasic() {
        assertEvaluatesTo(
            """
            main = 3 + 3
        """.trimIndent(), 6
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
    fun compileIf() {
        assertEvaluatesTo(
            """
            main = if (1 == 2) 1000 42
        """.trimIndent(), 42
        )

        assertEvaluatesTo(
            """
            main = if (1 < 2) 1000 42
        """.trimIndent(), 1000
        )
    }

    @Test
    fun compileFactorial() {
        assertEvaluatesTo(
            """
            fac n = if (n == 0) 1 (n * fac (n-1)) ;
            main = fac 3
        """.trimIndent(), 6
        )
    }

    @Test
    fun compilePair() {
        assertEvaluatesTo("""
            fst p = casePair p K ;
            snd p = casePair p K1 ;
            
            main = fst (snd (Pack{1,2} (1) (Pack{1,2} 3 4)))
        """.trimIndent(), 3)

        assertEvaluatesTo("""
            MkPair = Pack{1,2} ;
            fst p = casePair p K ;
            snd p = casePair p K1 ;
            
            main = fst (snd (fst (MkPair (MkPair 1 (MkPair 2 3)) 4)))
        """.trimIndent(), 2)
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

    @Test
    fun exampleArithmetic() {
        assertEvaluatesTo(
            """
            main = 4*5+(2-5)
        """.trimIndent(), 17
        )

        assertEvaluatesTo(
            """
            inc x = x+1;
            main = twice twice inc 4
        """.trimIndent(), 8
        )

        assertEvaluatesTo(
            """
            cons a b cc cn = cc a b ;
            nil      cc cn = cn ;
            length xs = xs length1 0 ;
            length1 x xs = 1 + (length xs) ;
            
            main = length (cons 3 (cons 3 (cons 3 nil)))
        """.trimIndent(), 3
        )
    }

    @Test
    fun exampleGcd() {
        assertEvaluatesTo("""
            gcd a b = if (a==b)
                        a
                      (if (a<b) (gcd b a) (gcd b (a-b))) ; 
            main = gcd 6 10
        """.trimIndent(), 2)
    }

    // Modified version of nfib, because nfib has a bug for nfib 2 (does not terminate).
    @Test
    fun fibonacci() {
        assertEvaluatesTo("""
            fib n = if (n <= 1) n (fib (n-1) + fib (n-2)) ;
            main = fib 7
        """.trimIndent(), 13)
    }

    /* TODO: Enable once list printing works
    @Test
    fun exampleListOutput() {
        assertEvaluatesTo("""
            cons = Pack{2,2} ;
            nil = Pack{1,0} ;
            
            downfrom n = if (n==0) nil (cons n (downfrom (n-1))) ;
            main = downfrom 4
        """.trimIndent(), 42)
    }
    */

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
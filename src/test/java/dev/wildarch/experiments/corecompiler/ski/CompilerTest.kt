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
}

private fun assertEvaluatesTo(program: String, num: Int): List<SkState> {
    val parsed = parse(program)
    val compiled = compile(parsed)
    val trace = evaluate(compiled)

    val finalState = trace.last()
    val finalValue = finalState.code
    assertThat(finalValue).isEqualTo(ConstInt(num))
    return trace
}

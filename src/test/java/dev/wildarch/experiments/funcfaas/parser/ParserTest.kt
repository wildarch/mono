package dev.wildarch.experiments.funcfaas.parser

import com.google.common.truth.Truth
import com.google.common.truth.Truth.assertThat
import dev.wildarch.experiments.funcfaas.syntax.*
import org.junit.Test

class ParserTest {
    @Test
    fun parseId() {
        assertThat(parseModule("""
            id :: a -> a;
            id x = x;
        """.trimIndent())).isEqualTo(Module(
            dataTypes = emptyList(),
            supercombs = listOf(
                Supercomb("id", TypeFun(TypeVar("a"), TypeVar("a")), listOf("x"), Var("x"))
            )
        ));
    }
}
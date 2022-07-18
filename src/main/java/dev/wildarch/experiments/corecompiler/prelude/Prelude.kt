package dev.wildarch.experiments.corecompiler.prelude

import dev.wildarch.experiments.corecompiler.parser.parse
import dev.wildarch.experiments.corecompiler.syntax.ScDefn

fun preludeDefs() : List<ScDefn> {
    return parse("""
        I x = x ;
        K x y = x ;
        K1 x y = y ;
        S f g x = f x (g x) ;
        compose f g x = f (g x) ;
        twice f = compose f f
    """.trimIndent()).defs
}
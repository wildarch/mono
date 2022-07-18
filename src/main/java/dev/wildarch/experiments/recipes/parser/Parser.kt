package dev.wildarch.experiments.recipes.parser

import dev.wildarch.corecompiler.RecipeBaseVisitor
import dev.wildarch.corecompiler.RecipeLexer
import dev.wildarch.corecompiler.RecipeParser
import org.antlr.v4.runtime.*
import org.antlr.v4.runtime.tree.ParseTree

sealed class RecipeNode

enum class MeasurementUnit {
    TEA_SPOON, TABLE_SPOON, GRAM, MILLILITER,
}

data class Quantity(val amount: Float, val unit: MeasurementUnit? = null) : RecipeNode()

data class Ingredient(val quantity: Quantity? = null, val name: String) : RecipeNode()

fun parseIngredient(source: String): Ingredient {
    // Boilerplate ANTLR setup
    val lexer = RecipeLexer(CharStreams.fromString(source))
    val tokens = CommonTokenStream(lexer)
    val parser = RecipeParser(tokens)

    // Install error listener
    parser.removeErrorListeners()
    val errorListener = ErrorListener()
    parser.addErrorListener(errorListener)

    // Run the parser and check for errors
    val tree: ParseTree = parser.ingredient()
    if (errorListener.errors.isNotEmpty()) {
        error("${errorListener.errors.size} parse errors: ${errorListener.errors.joinToString(separator = "\n")}")
    }

    // Transform to AST
    val astParser = Parser()
    return astParser.visit(tree) as Ingredient
}

class Parser : RecipeBaseVisitor<RecipeNode>() {
    override fun visitIngredient(ctx: RecipeParser.IngredientContext): Ingredient {
        return Ingredient(
            quantity = ctx.quantity()?.let(::visitQuantity),
            name = ctx.name().WORD().joinToString(separator = " ")
        )
    }

    override fun visitQuantity(ctx: RecipeParser.QuantityContext): Quantity {
        return Quantity(
            amount = ctx.NUM().text.toFloat(),
            unit =
            when (ctx.unit?.type) {
                RecipeParser.TSP -> MeasurementUnit.TEA_SPOON
                RecipeParser.TBSP -> MeasurementUnit.TABLE_SPOON
                RecipeParser.G -> MeasurementUnit.GRAM
                RecipeParser.ML -> MeasurementUnit.MILLILITER
                null -> null
                else -> error("Unknown unit: ${ctx.unit.text}")
            },
        )
    }
}

class ErrorListener : BaseErrorListener() {
    val errors = mutableListOf<String>()
    override fun syntaxError(
        recognizer: Recognizer<*, *>,
        offendingSymbol: Any,
        line: Int,
        charPositionInLine: Int,
        msg: String,
        e: RecognitionException?
    ) {
        val stack = (recognizer as RecipeParser).ruleInvocationStack.reversed()
        errors += "rule stack: $stack\nline $line:$charPositionInLine at $offendingSymbol: $msg"
    }
}
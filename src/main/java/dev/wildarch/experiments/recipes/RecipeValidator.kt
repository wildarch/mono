package dev.wildarch.experiments.recipes

import dev.wildarch.experiments.recipes.parser.parseIngredient
import org.commonmark.node.*
import org.commonmark.parser.Parser
import java.io.File
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    if (args.size != 1) {
        System.err.println("usage: recipe_validator RECIPE_PATH")
        exitProcess(1)
    }

    val mdParser = Parser.builder().build()
    val recipeFile = File(args[0])
    val recipe = mdParser.parseReader(recipeFile.bufferedReader())

    val ingredientsVisitor = IngredientsVisitor()
    recipe.accept(ingredientsVisitor)

    println("Raw ingredients found:\n${ingredientsVisitor.ingredients.joinToString(separator = "\n")}")

    val parsedIngredients = ingredientsVisitor.ingredients.map(::parseIngredient)
    if (parsedIngredients.isEmpty()) {
        System.err.println("error: no ingredients found")
        exitProcess(1)
    }
    println("Ingredients: \n${parsedIngredients.joinToString(separator = "\n")}")
}

class IngredientsVisitor : AbstractVisitor() {
    var ingredients = mutableListOf<String>()

    private var atIngredientsSection = false
    private var atIngredientsList = false
    private var atIngredientListItem = false


    override fun visit(heading: Heading) {
        // We are looking for a heading at level 2 called Ingredients
        // This allows for subsections under Ingredients
        if (heading.level <= 2) {
            // I am not sure if this is guaranteed to always be text, but I think it is a reasonable assumption for my
            // recipes.
            val text = heading.firstChild as Text
            atIngredientsSection = heading.level == 2 && text.literal == "Ingredients"
        }

        // No need to visit children here, we only care about the header text
    }

    override fun visit(bulletList: BulletList) {
        if (atIngredientsSection) {
            // A bullet list inside an ingredient section is a list of ingredients
            atIngredientsList = true
            visitChildren(bulletList)
            atIngredientsList = false
        }
    }

    override fun visit(listItem: ListItem?) {
        if (atIngredientsList) {
            atIngredientListItem = true
            visitChildren(listItem)
            atIngredientListItem = false
        }
    }

    override fun visit(text: Text) {
        if (atIngredientListItem) {
            ingredients.add(text.literal)
        }
    }
}
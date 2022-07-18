package dev.wildarch.experiments.recipes.parser

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class ParserTest {
    @Test
    fun parseLime() {
        val parsed = parseIngredient("1 lime")

        assertThat(parsed).isEqualTo(Ingredient(Quantity(1f), "lime"))
    }

    @Test
    fun parseCoconutMilk() {
        val parsed = parseIngredient("400ml coconut milk")

        assertThat(parsed).isEqualTo(Ingredient(Quantity(400f, MeasurementUnit.MILLILITER), "coconut milk"))
    }

    @Test
    fun hyphen() {
        val parsed = parseIngredient("thumb-sized piece of ginger")

        assertThat(parsed).isEqualTo(Ingredient(name = "thumb-sized piece of ginger"))
    }

    @Test
    fun parseJackFruitIngredients() {
        val ingredientsRaw = listOf(
            "1 red onion",
            "1 tsp ground cinnamon",
            "1 tsp cumin seeds",
            "2 tsp smoked paprika",
            "2 tsp chipotle hot sauce",
            "1 tbsp apple cider vinegar",
            "4 tbsp BBQ sauce",
            "200g can chopped tomato",
            "400g can jackfruit",
            "1 lime",
            "300g red cabbage",
            "1 cucumber",
            "7g fresh coriander",
            "4 tbsp mayonaise",
            "4 large wraps",
        );

        val parsedIngredients = ingredientsRaw.map(::parseIngredient)

        assertThat(parsedIngredients).containsExactly(
            Ingredient(Quantity(amount = 1f), name = "red onion"),
            Ingredient(Quantity(amount = 1f, unit = MeasurementUnit.TEA_SPOON), name = "ground cinnamon"),
            Ingredient(Quantity(amount = 1f, unit = MeasurementUnit.TEA_SPOON), name = "cumin seeds"),
            Ingredient(Quantity(amount = 2f, unit = MeasurementUnit.TEA_SPOON), name = "smoked paprika"),
            Ingredient(Quantity(amount = 2f, unit = MeasurementUnit.TEA_SPOON), name = "chipotle hot sauce"),
            Ingredient(Quantity(amount = 1f, unit = MeasurementUnit.TABLE_SPOON), name = "apple cider vinegar"),
            Ingredient(Quantity(amount = 4f, unit = MeasurementUnit.TABLE_SPOON), name = "BBQ sauce"),
            Ingredient(Quantity(amount = 200f, unit = MeasurementUnit.GRAM), name = "can chopped tomato"),
            Ingredient(Quantity(amount = 400f, unit = MeasurementUnit.GRAM), name = "can jackfruit"),
            Ingredient(Quantity(amount = 1f), name = "lime"),
            Ingredient(Quantity(amount = 300f, unit = MeasurementUnit.GRAM), name = "red cabbage"),
            Ingredient(Quantity(amount = 1f), name = "cucumber"),
            Ingredient(Quantity(amount = 7f, unit = MeasurementUnit.GRAM), name = "fresh coriander"),
            Ingredient(Quantity(amount = 4f, unit = MeasurementUnit.TABLE_SPOON), name = "mayonaise"),
            Ingredient(Quantity(amount = 4f), name = "large wraps"),
        )
    }
}
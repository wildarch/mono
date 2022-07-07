# Cooking recipes
A collection of my favourite recipes, accompanied with tools to manage them.

# Goal
There are lots of great recipes out there on the internet.
Unfortunately they are scattered across many different websites, and I tend to lose track of the recipes I find.
In addition to that I usually have to tweak them a little, and making them a second time requires me remembering what tweaks I applied.
Many recipe websites are also plain annoying:
- They are slow to load on my crappy phone
- Ads. Ads everywhere. Overlay videos are the worst offenders here.
- Long nonsense stories I have to scroll past to find the recipe

In addition to these gripes, I would love to be able to easily add all the ingredients for a recipe to my groceries app.
I do most of my shopping at Albert Heijn, having a button to add all the ingredients to the grocery list in their app would be awesome.
Unfortunately Albert Heijn has no free public API for this, so I may have to hack something together.

Thus, my goals are as follows:
- Collect my favourite recipes in a single place
- Allow full editing of ingredients and instructions, with option to link to an original recipe (by URL)
- Easy to browse on a mobile phone

And ideally I also want:
- Machine-readable ingredient list
- Mapping of ingredients to products available in my local grocery store
- One-click adding ingredients for a recipe to Albert Heijn grocery list

# Design
Things to figure out are:
- Where are recipes stored?
- In what format are they stored?
- How should we make the recipes accessible for viewing?
- How can we interact with Albert Heijn to add items to their grocery list?

## Storage
First and foremost, we need to store the recipes in a place that is durable. 
That means storing them locally is not an option.
Nothing about my recipes is private, so we can store them in this repository.
That makes them very easy to access and share. 
We also get a history of changes to the recipe for free, which is a nice bonus.
The one caveat here is the recipes are probably not very easy to edit on a mobile device, but I don't really do that anyway.
Another option would be to put them in Google Drive. 
That would also work, but is not as easy to access programmatically.

**Decision**: Store in git.

## Format
This is probably the most difficult and important question to answer.
We probably want a textual representation, which both easier to edit and track in git.

The simplest option is free-form flat text. I think that is a bit too simple for a recipe, as they are usually quite structured:
1. List of ingredients, one by one
2. Instructions, step by step

So, we want something with a bit more structure.

HTML and XML are very structured, but a bit too much for my taste. Markdown could be a good compromise? 
It is easy to write, and GitHub can render it natively. 
One problem there is that it might not be structured enough to map to Albert Heijn products.
We could make ingredients links to product pages, but that is tedious to write.
I don't think it is very easy to add annotations to a markdown document that are not normally rendered.
Maybe I can get away with it if I force myself to write ingredients down in a very structured way, and write a simple parser?

There are actually many open formats for recipes that I found with a quick search:
- [Open Recipe Format](https://open-recipe-format.readthedocs.io/en/latest/). YAML based format that looks quite verbose. I think it is just a schema, can't find any tooling or public users. They have field for the *recipe yield*, which seems like a smart thing to have. **skip**.
- [Google Recipe Schema Markup](https://developers.google.com/search/docs/advanced/structured-data/recipe). Used for SEO so Google can render your recipe for you in search results. Ingredients are not structured enough for our purposes. I don't plan to include these recipes in search results, **skip**.
- [schema.org Recipe](https://schema.org/Recipe). schema.org seems like a pretty big organisation, which is promising. Looks like it is intended to be embedded into webpages to make them machine-readable. It looks tedious to edit by hand. Their format is highly structured, I might steal some bits from it. But mostly, **skip**.
- [Microformats overview of different formats](https://microformats.org/wiki/recipe-formats). All of this looks pretty ancient. I did find a link to a website that sounds cool: http://www.cookingforengineers.com/. For all the projects listed, **skip**.
- [Open Recipes](https://openrecip.es/). Looks like a collection links to recipes. Honestly I am not sure what this really is, but it does not look useful to me. **skip**.

I conclude from this that there is no suitable existing format, so I will have to roll my own.
To keep things simple for now, **I am going to put the recipes in markdown format**.
This requires no extra work to view the recipes on any device with a browser, since GitHub will render them for me.

## Machine-readable ingredients list
My recipes are stored in markdown without a strict schema, so parsing them may be a bit tricky.
Hopefully if I am consistent in how I write the recipes, it will be quite easy to work with.
It helps a lot that I will only be parsing recipes I author myself.

I have some experience with ANTLR, so I would like to use that to parse out the ingredients into a more structured form, something like:

```
'2 tbsp sugar' -> ingredient {
    quantity: 2,                    // Optional if we do not have a specific amount, e.g. 'salt'
    unit: TABLE_SPOON,              // Optional if there is a natural unit, e.g. '1 lime'
    name: "sugar",
}
```

So far I have only built a prototype, but it seems work fairly well.
The one problem is that if a unit is not understood, it will consider it part of the name.
Hopefully I can do better error detection on that once I have a database of valid names.

Unfortunately it seems I cannot reliably parse full recipes with ANTLR, because [it can only handle context-free grammars](https://github.com/antlr/grammars-v4/issues/472).
Instead we can extract a list of ingredients with a dedicated markdown parser, and parse individual ingredients using ANTLR. That may actually be better than parsing everything in ANTLR, because it means I have less grammar to maintain.

The current grammar is used to check that we can parse the ingredients of all recipes, but since the grammar is very lenient, I do not think it actually validates much, besides checking that an ingredients section exists.

## Mapping to product available at AH
For this we need some kind of database or lookup table with valid product names.
Just having a list of valid product names will make parsing much more reliable.
It will also make parsing more strict, hopefully it won't be too annoying when adding new recipes.

Later on we will also need to store what unit the product is measured in, and in what quantities it is available.
That will allow us to create actual grocery lists.
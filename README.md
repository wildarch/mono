# mono
My own personal monorepo.

This code is mostly just my experiments and little tools I write for my own use.
Still, you might find something here that is useful to you (as an example).
I list a few potentially interesting bits and bobs below.

## Recipes
Take a look at my [recipe collection](https://github.com/wildarch/mono/tree/main/recipes/collections).
If you are curious why I store them here, why I use markdown and what other cool stuff I plan to do with them eventually, read more [here](https://github.com/wildarch/mono/blob/main/recipes/README.md).

## Ansible playbooks
I have Ansible playbooks to provision:
- A build environment to successfully build this repository, and run the tests.
- My development machine `yoga`.
- A Raspberry Pi used as a Spotify Connect speaker.

## Compiler for a lazy functional programming language
Based on [*Implementing functional languages: a tutorial*](https://www.microsoft.com/en-us/research/publication/implementing-functional-languages-a-tutorial/), I am implementing a series of compilers for a small lazy functional language, to learn more about how they are built.
Hopefully I will eventually get to implementing a spineless tagless G-machine, like [Haskell's GHC](https://gitlab.haskell.org/ghc/ghc/-/wikis/commentary/compiler/generated-code).
Code is available [here](https://github.com/wildarch/mono/tree/main/src/main/java/dev/wildarch/experiments/corecompiler).
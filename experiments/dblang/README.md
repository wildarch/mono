# DBlang
A low-level systems programming language with first-class reflection support, with the goal of easily generating and compiling code at runtime.

## Syntax
### Definitions
We need the following definitions at the top-level:
- `struct`
- `enum`
- `alias` Type alias
- `func` Function
- `const` Constant
- `global` Global variable

I want to design the language so that it can be parsed very efficiently.
One idea to experiment with is a multi-pass parse:
1. Split the input file into per-definition chunks (recording start and end offsets per definition).
   Detect the start of a definition from the `def` keyword (this keyword is not allowed anywhere else). 
   Every definition has the format `def <name> = <defkind> { .. }
2. Parse individual definitions.
   This needs to leave placeholders for references to other definitions.

An additional benefit is that we can scope error messages regarding syntax much better.
Example syntax:

```
def MyStruct = struct {
    field1: T1,
    field2: T2,
}
def MyEnum = enum {
    OPTION_ONE,
    OPTION_TWO,
}
def MyAlias = alias { <type expr> }
def MyFunc = func(p1: T1, p2: T2) -> T3 { ... }
def MyConst = const { <expr> }
def MyGlob = global { <type expr> }
```

Then for type checking:
1. Build a dependency graph for definitions that reference each other.
2. Create an ordering over the dependency graph.
   Can partition where disjoint (for parallel execution), topological ordering within a partition.
3. Perform the checks.
   Some might not be possible if a function has an abstract type.
   This is fine: type checking in this case is (partially?) postponed until the function is specialized.

### Types

### Statements
TODO

### Expressions
# IR Operations
The set of IR operations needed to represent everything defined by [syntax.md](syntax.md).

The design follows the existing IR in `ir/`: SSA values, structured control flow with regions, and memory-based locals where lvalues are addresses.
Ops in **bold** are missing from `OpKind` in `ir/Operation.h`.

## Values and Constants
| Op | Covers |
|---|---|
| `CONST_INT` (value + type) | `int_lit`, `char_lit`, `true`/`false`, enum values (an integer once resolved), folded `sizeof`/`alignof` |
| **`CONST_FLOAT`** | `float_lit` |
| **`CONST_STRING`** | `string_lit`: returns the address of static byte data |
| **`GLOBAL_ADDR`** (def) | a `DefRef` that names a `global`; reads and writes go through `LOAD`/`STORE` |
| `FUNC_ADDR` (def) | a `DefRef` that names a `func` used as a value, e.g. assigned to a `func(...)`-typed variable |
| **`CONST_REF`** (def), or inline the const's region | a `DefRef` that names a `const` |
| **`SIZEOF` / `ALIGNOF`** (type) | only needed if types can still be abstract or unresolved when the IR is built; otherwise fold to `CONST_INT` |

A local `DefRef` or parameter produces no op.
It refers to the variable's `ALLOCA` result, or to the function's entry block argument.

## Memory and Addresses (lvalues)
| Op | Covers |
|---|---|
| `ALLOCA` (type) | each `let` and each spilled parameter; temporaries for struct and array literals |
| `LOAD` / `STORE` | reading an lvalue, `=`, the read-modify-write of compound assignment, `++`/`--`, unary `*` in rvalue position. Must also work on aggregate types for struct/array assignment. |
| `FIELD_PTR` (ptr, field index) | `e.f` on an lvalue; `e->f` is `FIELD_PTR(LOAD e)` |
| **`ELEM_PTR`** (ptr, index) | `e[i]` on an array lvalue or a pointer (`p[i]` ≡ `*(p+i)`); pointer `+`/`-` integer |
| **`PTR_DIFF`** | `p - q` |
| **`EXTRACT_FIELD` / `EXTRACT_ELEM`**, or spill to a temporary `ALLOCA` | `.f` or `[i]` on an rvalue aggregate, e.g. `make_point().x` or `[1, 2, 3][i]` |

Unary `&` produces no op, because the lvalue already is an address.

## Aggregate Construction
| Op | Covers |
|---|---|
| **`STRUCT_CREATE`** (n operands), or `ALLOCA` + n× `FIELD_PTR`/`STORE` | `S { x: e1, y: e2 }` |
| **`ARRAY_CREATE`** (n operands) | `[a, b, c]` |
| **`ARRAY_SPLAT`** (value, constant count) | `[e; N]` |
| **`UNDEF`** or **`ZERO`** | only if `S {}` or a partial field list is allowed (unspecified) |

## Arithmetic, Bitwise and Comparison
Signedness is decided either by the operand type or by separate ops; pick one convention.

- **`NEG`** (unary `-`) and **`BIT_NOT`** (`~`).
  Unary `+` produces no op, apart from integer promotion.
  `!` becomes `CMP eq 0`.
- `MUL`, `DIV`, `MODULO` (`*` `/` `%`).
  Signed vs unsigned matters for `DIV`/`MODULO`, and float vs int matters too.
- `ADD`, `SUB`, `SHIFT_LEFT`, `SHIFT_RIGHT` (arithmetic vs logical shift, decided by type).
- `CMP` with a predicate: `eq ne lt le gt ge`, plus signed, unsigned or float.
  It must also accept pointers.
- `AND`, `XOR`, `OR` as **bitwise** ops only.
  `&&` and `||` must not lower to these (see [Calls and Expression-Level Control Flow](#calls-and-expression-level-control-flow)).
- `CAST` (from type → to type): one op for explicit `cast<T>(e)` and every implicit conversion that C semantics bring in (integer promotions, usual arithmetic conversions, `if (ptr)` truth tests).
  It can be split later into trunc/ext/int↔float/ptr↔int.

## Calls and Expression-Level Control Flow
| Op | Covers |
|---|---|
| `CALL` (direct, def) | `f(args)`, `c::f(args)`, `S::f(args)`; `x.f(args)` with `x` (or `&x`) as the first argument |
| **`CALL_INDIRECT`** (callee value) | calling an expression of type `func(...)` |
| `IF` **with results** + **`YIELD`** terminator | `a ? b : c`, `a && b`, `a \|\| b` |

`&&`, `||` and `?:` must short-circuit because operands can be calls, so a `SELECT` op is not enough to express them.

## Statements and Structured Control Flow
| Op | Covers |
|---|---|
| `IF` (cond; then-region, else-region) | `if`/`else`, `else if` (nested `IF` inside the else-region) |
| `LOOP` with separate **cond**, **body** and **step** regions, plus a "test first / test last" flag | `while`, `do … while`, `for` |
| **`SWITCH`** (scrutinee; ordered case regions, each with a list of constant values or `default`) | `switch` |
| **`FALLTHROUGH`** terminator, or implicit fallthrough at the end of a case region | C fallthrough |
| `BREAK` (**target**: loop or switch) | `break` |
| **`CONTINUE`** (target loop) | `continue` |
| `RETURN` (0 or 1 operand) | `return`, `return e` |
| **`YIELD`** | terminator for value-producing regions (the loop cond, `IF` with results) |

Non-obvious requirements:
- **Loops need a separate step region** because `continue` in a `for` must still run the step clause.
  A single `LOOP { body }` with an `IF … BREAK` at the top would skip the step.
- **`BREAK` and `CONTINUE` need an explicit target.**
  A `break` can sit inside an `if` inside a `case`, and a `continue` inside a `switch` targets the enclosing loop.
  "Innermost enclosing" is not the same construct for both: `break` exits the `switch`, `continue` goes to the loop.
- **Fallthrough means `SWITCH` cannot simply be a chain of `IF`s.**
  Use ordered case regions with fallthrough, or let `SWITCH` be the one place that allows a small unstructured CFG.

## Definition Level (not ops)
- `func`: a region whose entry block takes the parameters as block arguments.
- `const`: a region that yields one value and is evaluated at compile time.
- `global`: a type only, with no initializer, matching the spec.
- `struct`, `enum` and `alias`: type-level, already in `Def.h`.

Several places need a **constant evaluator** over this IR (or over the AST): enum `= Expr`, `[T; Expr]`, the count in `[e; N]`, and `case` labels.

## Open Questions in syntax.md
These affect the op set:
1. **`float_lit` has no type.**
   `TypePrimitive` has no `f32`/`f64`, and neither does `TypeKind`, though the comment in `TypeImpl.h` mentions them.
   Either add float types (and float variants of the arithmetic, `CMP` and `CAST` ops) or drop `float_lit`.
2. **The type of `string_lit` is undefined.**
   It could be `*char`, `*u8` or `[char; N]`, and the answer decides what `CONST_STRING` returns.
3. **There is no null pointer literal.**
   The options are `cast<*T>(0)` (int→ptr `CAST` on a constant) or a dedicated literal.
4. **`let` scoping inside `switch` cases is unclear.**
   `SwitchCase` holds bare `Stmt*`, not a `Block`.
   A `let` in `case 1` could be visible but uninitialized in `case 2` when control jumps straight to `case 2`, as in C.
   Either scope each case or forbid this.
5. **Member-call receiver passing is unspecified.**
   Is `x.f()` passed `x` by value or `&x`? What does `p->f()` mean?
6. **Omitted struct fields are unspecified.**
   This determines whether `UNDEF`/`ZERO` is needed.

## Changes to the Current `OpKind`
- Add: `CONTINUE`, `SWITCH`, `YIELD`, `CALL_INDIRECT`, `ELEM_PTR`, `PTR_DIFF`, `GLOBAL_ADDR`, `CONST_STRING`, `CONST_FLOAT`, `NEG`, `BIT_NOT`, aggregate construction ops.
- Change: `IF` gets results, `BREAK` gets a target, `AND`/`OR` are explicitly bitwise.

## Extra Data
Some ops require data beyond a result type and operands:
- References to definitions (GLOBAL_ADDR, FUNC_ADDR, CALL)
- Type (CAST, SIZEOF, ALIGNOF)
- Indices (FIELD_PTR, EXTRACT_FIELD, ARRAY_SPLAT)
- Flags (CMP)
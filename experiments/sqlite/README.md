# SQLite C Language Subset Analysis

This document describes the subset of the C language that must be implemented
in order to correctly parse `sqlite3.c`, as determined by analysing the
Intermediate Representation (IR) constructed by `parse_to_ir.py`.

The parser uses `pycparser` to build an AST from the preprocessed C file,
then converts it into a custom JSON-serializable IR. Every IR node kind
corresponds to a C language construct that appears in `sqlite3.c`. The
sections below enumerate all constructs that the IR handles, organised by
category, and then list the most notable C features that are **not**
required.

---

## Required C Constructs

### 1. Top-Level Declarations

| IR Kind | C Construct |
|---------|-------------|
| `File` | Translation unit (the entire file) |
| `FuncDef` | Function definition (with body) |
| `Decl` | Global variable declaration, function declaration (prototype), or variable declaration with optional initializer |
| `TypeDef` | `typedef` declaration |
| `Pragma` | `#pragma` directive (passed through as opaque string) |

### 2. Statements

| IR Kind | C Construct |
|---------|-------------|
| `Compound` | Block `{ ... }` (compound statement) |
| `If` | `if` / `if`-`else` |
| `While` | `while` loop |
| `DoWhile` | `do`-`while` loop |
| `For` | `for` loop (including declaration in the init clause via `DeclList`) |
| `Switch` | `switch` statement |
| `Case` | `case` label |
| `Default` | `default` label |
| `Return` | `return` statement |
| `Break` | `break` statement |
| `Continue` | `continue` statement |
| `Goto` | `goto` statement |
| `Label` | Named label (label `:` statement) |
| `EmptyStatement` | Empty statement (`;`) |
| `Decl` | Declaration used as a statement (C99 mixed declarations) |
| `Typedef` | `typedef` inside a function body (C99) |
| `DeclList` | Comma-separated declaration list (used in `for`-loop init) |

Any expression node may also appear as a statement (expression statement).

### 3. Expressions

| IR Kind | C Construct |
|---------|-------------|
| `Constant` | Integer, float, character, and string literals |
| `ID` | Identifier reference |
| `BinaryOp` | Binary operators: `+`, `-`, `*`, `/`, `%`, `<<`, `>>`, `&`, `|`, `^`, `&&`, `||`, `==`, `!=`, `<`, `>`, `<=`, `>=` |
| `UnaryOp` | Unary operators: `++` (prefix/postfix), `--` (prefix/postfix), `+`, `-`, `!`, `~`, `*` (dereference), `&` (address-of) |
| `TernaryOp` | Ternary conditional `?:` |
| `Assignment` | Assignment: `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `<<=`, `>>=`, `&=`, `|=`, `^=` |
| `FuncCall` | Function call |
| `Cast` | Explicit type cast `(type) expr` |
| `Sizeof` | `sizeof` operator (applied to both expressions and types) |
| `ArrayRef` | Array subscript `a[i]` |
| `StructRef` (dot) | Struct member access `a.b` |
| `StructRef` (arrow) | Struct pointer member access `a->b` |
| `CompoundLiteral` | C99 compound literal `(type){ ... }` |
| `InitList` | Initializer list `{ e1, e2, ... }` |
| `NamedInitializer` | C99 designated initializer `.field = value` |
| `ExprList` | Comma expression / expression list |

### 4. Types

| IR Kind | C Construct |
|---------|-------------|
| `IdentifierType` | Built-in types (`int`, `char`, `void`, `float`, `double`, `short`, `long`, `signed`, `unsigned`) and any other named type |
| `TypeDecl` | Type declaration node (wraps an `IdentifierType` or derived type with a declarator name and qualifiers) |
| `PtrDecl` | Pointer type `*` (with optional `const`/`volatile`/`restrict` qualifiers) |
| `ArrayDecl` | Array type (with optional dimension expression) |
| `FuncDecl` | Function type (return type + parameter list) |
| `Struct` | `struct` definition (with member list) |
| `Union` | `union` definition (with member list) |
| `Enum` | `enum` definition (with named values) |
| `Enumerator` | Individual `enum` value |
| `Typename` | Type name (used in casts and `sizeof(type)`) |

### 5. Declarations and Parameters

- **Variable declarations** with optional initializer, optional storage-class
  specifiers (`static`, `extern`, `register`, `auto`, `typedef`), and optional
  bit-field size (`bitsize`).
- **Function declarations** (prototypes) with parameter lists.
- **Function definitions** with a body (compound statement).
- **Parameter lists** (`ParamList`) including:
  - Named parameters (via `Decl`)
  - Unnamed parameters (via `Typename`)
  - Variadic parameter (`...` / `EllipsisParam`)

### 6. Preprocessor Handling

The input file is run through the C preprocessor (`cpp -P`) before parsing.
This means the following are handled **before** the C parser sees the code:

- `#include` directives (resolved to the `pycparser` fake libc headers)
- `#define` / `#undef` macro definitions and expansions
- `#if` / `#ifdef` / `#ifndef` / `#else` / `#elif` / `#endif` conditional compilation
- `#pragma` directives (preserved as `Pragma` nodes in the IR)
- `__attribute__` annotations (stripped via `-D__attribute__(x)=`)

---

## Notable C Features NOT Required

The following standard C features are either **not handled** by `parse_to_ir.py`
or are **not present** in `sqlite3.c` after preprocessing. A parser targeting
only `sqlite3.c` can safely omit these:

### Preprocessor
- **Macro handling** — All preprocessing is done by `cpp` before parsing. The
  parser never sees `#define`, `#include`, `#if`, etc. Function-like macros
  (e.g. `MAX(a,b)`) are fully expanded and do not appear in the IR.

### Declarations
- **`register` storage class** — While the parser captures storage-class
  specifiers generically, `register` is deprecated in C11 and rarely used in
  modern code.
- **`auto` storage class** — Implicit for local variables; never explicitly used
  in `sqlite3.c`.
- **Inline assembly (`asm` / `__asm__`)** — Not used in `sqlite3.c`. Handled
  by the preprocessor or compiler-specific extensions.
- **`_Thread_local` / `thread_local`** — C11 thread-local storage; not used.

### Types
- **`_Bool` / `bool`** — SQLite uses `int` for boolean values.
- **`_Complex` / `_Imaginary`** — C99 complex number types; not used.
- **`long double`** — Not used in `sqlite3.c`.
- **Variable-length arrays (VLAs)** — C99 `int a[n]` where `n` is runtime;
  `sqlite3.c` uses fixed-size arrays or pointers.
- **Flexible array members** — C99 `struct { int n; int arr[]; }`; not used.
- **`_Atomic`** — C11 atomic types; not used (SQLite has its own locking).
- **Type qualifiers beyond `const`/`volatile`/`restrict`** — The parser
  captures qualifiers generically, but `_Atomic` and `__attribute__`-based
  qualifiers are not needed.

### Expressions
- **Compound assignment on complex types** — Not applicable without complex types.
- **`_Generic`** — C11 generic selection; not used.
- **Statements expressions `({ ... })`** — GNU extension; not used in `sqlite3.c`.

### Statements
- **`_Static_assert`** — C11 static assertions; handled by preprocessor/compiler.
- **Range-based `case` labels (`case 1 ... 5`)** — GNU extension; not used.

### Standard Library
- **Standard library headers** — All `#include` directives are resolved to
  `pycparser`'s fake libc headers, which provide minimal declarations. The
  actual library implementations are not parsed.

---

## Summary

To parse `sqlite3.c`, a C parser must handle:

1. **The core expression grammar**: all arithmetic, logical, bitwise,
   comparison, assignment, ternary, `sizeof`, cast, function call, array
   subscript, and struct/union member access operators.
2. **All statement types**: compound blocks, `if`/`else`, `while`,
   `do`-`while`, `for`, `switch`/`case`/`default`, `return`, `break`,
   `continue`, `goto`, labels, and expression statements.
3. **The type system**: built-in types, pointers, arrays, functions,
   `struct`, `union`, `enum`, `typedef`, and type qualifiers
   (`const`, `volatile`, `restrict`).
4. **Declarations**: global and local variable declarations, function
   declarations and definitions, parameter lists (including variadic), and
   initializers (including designated initializers and compound literals).
5. **Preprocessor integration**: the file must be preprocessed with `cpp`
  first to resolve macros, includes, and conditional compilation.

The parser does **not** need to handle complex numbers, VLAs, `_Atomic`,
`_Generic`, inline assembly, or any feature that is either absent from
`sqlite3.c` or resolved by the preprocessor before parsing.

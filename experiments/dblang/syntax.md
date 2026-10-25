# DBLang Syntax
Details the grammar of DBLang.

## Source Files
A DBLang program is a collection of source files that are linked together.

```
SourceFile = Import* Def*;

Import = "import" ImportPath;
ImportPath = string_lit;
```

Import paths are file system paths with the `.dblang` extension stripped, relative to the project root.
Example: to reference `MyDef` defined in source file `a/b/c.dblang`, write:

```
import "a/b/c"

def MyAlias = alias c::MyDef
```

## Definitions
A definition starts with the keyword `def`:

```
Def = "def" (ident "::")? ident "=" AnyDef;
AnyDef = StructDef | EnumDef | AliasDef | FuncDef | ConstDef | GlobalDef;
```

Most definitions use a single identifier, the only exception being member functions of struct or enum types.

A reference to a definition is a path of identifiers separated by `::`:

```
DefRef = ident ("::" ident)*;
```

The path can start with the name of an imported file, and can end with an enum value or member function of a type.
Which definition a path refers to is resolved after parsing.

### Struct
A struct is a user-defined type with named fields.

```
StructDef = "struct" "{" StructFields? "}";
StructFields = StructField ("," StructField)* ","?;
StructField = ident ":" Type;
```

### Enum
A named set of integer constants.

```
EnumDef = "enum" "{" EnumAlt ("," EnumAlt)* ","? "}";
EnumAlt = ident ("=" Expr)?;
```

### Alias
Make an existing definition available under a different name.

```
AliasDef = "alias" DefRef;
```

### Function
Functions follow a Rust-like syntax:

```
FuncDef = "func" "(" FuncParams? ")" ("->" Type)? Block; 
FuncParams = FuncParam ("," FuncParam)* ","?;
FuncParam = ident ":" Type;
```

Function defs whose name consists of two parts indicate a member function.
For example, a function `def MyStruct::my_func = func(..) { ... }` is a member of `MyStruct`, and can be called as `myStructInstance.my_func()`.

### Const
A compile-time constant value.

```
ConstDef = "const" "{" Expr "}";
```

### Global
A global mutable variable.

```
GlobalDef = "global" "{" Type "}";
```

The value of a global variable before it is first assigned to is undefined.

### Type
```
Type = TypePrimitive | TypeDef | TypeArray | TypePointer | TypeFunction;
TypePrimitive = "void"
              | "bool" 
              | "char" 
              | "i8" 
              | "i16"
              | "i32"
              | "i64"
              | "i128"
              | "isize"
              | "u8" 
              | "u16"
              | "u32"
              | "u64"
              | "u128"
              | "usize";
TypeDef = DefRef;
TypeArray = "[" Type ";" Expr "]";
TypePointer = "*" Type;
TypeFunction = "func" "(" FuncParams? ")" ("->" Type)?;
```

### Identifier
An identifier is a C-style identifier.
The first character must be a nondigit (from `a` to `z`, `A` to `Z` or `_`), subsequent characters may include digits `0` to `9`.

### Block
A block is a collection of statements.

```
Block = "{" Stmt* "}";
```

### Statement
Statements follow C (C23 §6.8), with the differences listed below.

```
Stmt = SimpleStmt ";"
     | Block
     | IfStmt
     | SwitchStmt
     | WhileStmt
     | DoWhileStmt
     | ForStmt
     | "return" Expr? ";"
     | "break" ";"
     | "continue" ";";

SimpleStmt = LetDecl
           | Expr
           | Expr AssignOp Expr
           | Expr ("++" | "--");
AssignOp = "=" | "*=" | "/=" | "%=" | "+=" | "-=" | "<<=" | ">>=" | "&=" | "^=" | "|=";
LetDecl = "let" ident (":" Type)? "=" Expr;

IfStmt = "if" "(" Expr ")" Block ("else" (IfStmt | Block))?;

SwitchStmt = "switch" "(" Expr ")" "{" SwitchCase* "}";
SwitchCase = SwitchLabel+ Stmt*;
SwitchLabel = "case" Expr ":" | "default" ":";

WhileStmt = "while" "(" Expr ")" Block;
DoWhileStmt = "do" Block "while" "(" Expr ")" ";";
ForStmt = "for" "(" SimpleStmt? ";" Expr? ";" SimpleStmt? ")" Block;
```

Assignment, increment and decrement are statements, not expressions, so they cannot be chained or nested: `a = b = c;` and `a[i++] = 0;` are syntax errors.
The operand being modified must be an lvalue, which is checked after parsing.

Any expression can be used as a statement, e.g. a function call `foo();`.
There is no null statement: an empty body is written `{}`.

A `let` declaration introduces a local variable that is in scope until the end of the enclosing block.
The initializer is required; if the type is omitted, it is the type of the initializer.

The bodies of `if`, `else`, `switch`, `while`, `do` and `for` are always blocks, so there is no dangling `else`.
`else if` is allowed without braces around the nested `if`.

Labels only exist as `case` and `default` labels directly inside a `switch`, and there is no `goto`.
As in C, execution falls through from one case to the next unless it reaches a `break`.
Case expressions must be constant, which is checked after parsing.

A `let` in the first clause of a `for` is scoped to the loop.

### Expression
Expressions follow the syntax of C (C23 §6.5), with these differences:
- There are no comma, assignment, increment or decrement operators.
- `sizeof` only takes a type.
- Casts are written `cast<T>(e)`.
- Struct instances are created with `S { x: e1, y: e2 }` instead of compound literals.

```
Expr = DefRef
     | Literal
     | "(" Expr ")"
     | DefRef "{" FieldInits? "}"
     | "[" ArrayElems? "]"
     | "[" Expr ";" Expr "]"
     | "cast" "<" Type ">" "(" Expr ")"
     | "sizeof" "(" Type ")"
     | "alignof" "(" Type ")"
     | Expr "[" Expr "]"                          [assoc=left, prec=13]
     | Expr "(" CallArgs? ")"                     [assoc=left, prec=13]
     | Expr ("." | "->") ident                    [assoc=left, prec=13]
     | ("&" | "*" | "+" | "-" | "~" | "!") Expr   [assoc=right, prec=12]
     | Expr ("*" | "/" | "%") Expr                [assoc=left, prec=11]
     | Expr ("+" | "-") Expr                      [assoc=left, prec=10]
     | Expr ("<<" | ">>") Expr                    [assoc=left, prec=9]
     | Expr ("<" | ">" | "<=" | ">=") Expr        [assoc=left, prec=8]
     | Expr ("==" | "!=") Expr                    [assoc=left, prec=7]
     | Expr "&" Expr                              [assoc=left, prec=6]
     | Expr "^" Expr                              [assoc=left, prec=5]
     | Expr "|" Expr                              [assoc=left, prec=4]
     | Expr "&&" Expr                             [assoc=left, prec=3]
     | Expr "||" Expr                             [assoc=left, prec=2]
     | Expr "?" Expr ":" Expr                     [assoc=right, prec=1];

Literal = int_lit | float_lit | char_lit | string_lit | "true" | "false";
CallArgs = Expr ("," Expr)*;

FieldInits = FieldInit ("," FieldInit)* ","?;
FieldInit = ident ":" Expr;
ArrayElems = Expr ("," Expr)* ","?;
```

A `DefRef` expression names one of:
- A local variable or function parameter, e.g. `x`.
- A function, constant or global in the same file, e.g. `my_func` or `MY_CONST`.
- A definition from an imported file, e.g. `c::my_func`.
- An enum value, e.g. `MyEnum::MY_ALT`, or `c::MyEnum::MY_ALT` if the enum is defined in an imported file.
- A member function, e.g. `MyStruct::my_func` or `c::MyStruct::my_func`.

In a struct literal, the `DefRef` names the struct type, e.g. `c::Point { x: 1, y: 2 }`.

Precedence and associativity:
- A higher `prec` binds tighter. Alternatives without a `prec` are atoms and bind tightest of all.
- An operand of an operator cannot be an unparenthesized expression with a lower `prec`.
  This only restricts operands that are not already enclosed by the operator's own tokens: the operands inside `[ ]`, `( )` and between `?` and `:` can be any expression.
- When an operand has the same `prec` as its operator, `assoc` decides the grouping: `a - b - c` is `(a - b) - c` and `a ? b : c ? d : e` is `a ? b : (c ? d : e)`.

Unlike C, there is no `_Generic`, and `alignof` uses the C23 spelling.

Compound literals use Rust syntax instead of C's `(T){ ... }`:
- A struct literal names the struct type followed by its fields: `Point { x: 1, y: 2 }`.
- An array literal lists its elements, `[1, 2, 3]`, or repeats one value a constant number of times, `[0; 16]`.
  The element type is inferred from the elements.

There is no ambiguity between a struct literal and a block, because the conditions of `if`, `switch`, `while` and `for` are parenthesized.

### TODO
int_lit, float_lit, char_lit, string_lit
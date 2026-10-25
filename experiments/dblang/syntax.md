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
AliasDef = "alias" (ident "::")? ident;
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
Type = TypePrimitive | TypeDef | TypeArray | TypePtr | TypeFunc;
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
TypeDef = (ident "::") ident;
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
Block = "{" Stmt* "}"
```

### Expr
TODO: All expressions as they are included in C23.

### Statement
TODO: this section is a stub.

```
Stmt = AssignStmt | IncDecStmt;
AssignStmt = Expr AssignOp Expr ";";
AssignOp = "=" | "*=" | "/=" | "%=" | "+=" | "-=" | "<<=" | ">>=" | "&=" | "^=" | "|=";
IncDecStmt = Expr ("++" | "--") ";";
```

Assignment, increment and decrement are statements, not expressions, so they cannot be chained or nested: `a = b = c;` and `a[i++] = 0;` are syntax errors.
The operand being modified must be an lvalue, which is checked after parsing.

### Expression
Expressions follow the syntax of C (C23 §6.5), except that there are no comma, assignment, increment or decrement operators, `sizeof` only takes a type, and casts are written `cast<T>(e)`.

```
Expr = (ident "::")? ident
     | Literal
     | "(" Expr ")"
     | "(" Type ")" "{" InitList? "}"
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

InitList = Initializer ("," Initializer)* ","?;
Initializer = Designation? (Expr | "{" InitList? "}");
Designation = Designator+ "=";
Designator = "[" Expr "]" | "." ident;
```

Precedence and associativity:
- A higher `prec` binds tighter. Alternatives without a `prec` are atoms and bind tightest of all.
- An operand of an operator cannot be an unparenthesized expression with a lower `prec`.
  This only restricts operands that are not already enclosed by the operator's own tokens: the operands inside `[ ]`, `( )` and between `?` and `:` can be any expression.
- When an operand has the same `prec` as its operator, `assoc` decides the grouping: `a - b - c` is `(a - b) - c` and `a ? b : c ? d : e` is `a ? b : (c ? d : e)`.

Unlike C, there is no `_Generic`, and `alignof` uses the C23 spelling.

### TODO
Stmt, int_lit, float_lit, char_lit, string_lit
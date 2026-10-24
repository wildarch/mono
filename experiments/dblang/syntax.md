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
Def = "def" ident "=" AnyDef;
AnyDef = StructDef | EnumDef | AliasDef | FuncDef | ConstDef | GlobalDef;
```

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
A function can be a member function of a struct type, or it can be a 'free' function:

```
FuncDef = "func" "(" FuncParams? ")" ("->" Type)? Block; 
FuncParams = FuncParam ("," FuncParam)* ","?;
FuncParam = ident ":" Type;
```

If the first parameter is named `this`, the function is a member function.
For example, a function `def my_func = func(this: *MyStruct, ...) { ... }` is a member of `MyStruct`, and can be called as `myStructInstance.my_func()`.

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

### TODO
ident, Block, Expr, string_lit
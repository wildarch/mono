#pragma once

#include <cstddef>
#include <span>
namespace dblang::ir {

struct Def;
struct DefStruct;
struct DefEnum;

class Type {
public:
  enum BasicType : unsigned {
    UNRESOLVED,
    VOID,
    // Primitive types
    BOOL,
    CHAR,
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    ISIZE,
    USIZE,
    F32,
    F64,
    // Compound/nested types
    STRUCT,
    ENUM,
    ARRAY,    // fixed-size array
    FUNCTION, // function reference
  };

  class ExtraInfo;

private:
  ExtraInfo *extraInfo = nullptr;
  BasicType _basicType = UNRESOLVED;
  unsigned _pointerDepth = 0;

public:
  explicit constexpr Type(BasicType basicType) : _basicType(basicType) {}
  static Type forStruct(const DefStruct &def);
  static Type forEnum(const DefEnum &def);
  static Type forArray(Type elemType, std::size_t size);
  static Type forFunction(std::span<const Type> paramTypes, Type returnType);

  // Pointer
  Type getPointerTo();
  Type getPointee();

  // Struct
  const DefStruct &getStruct() const;

  // Enum
  const DefEnum &getEnum() const;

  // Array
  const Type &getArrayElementType() const;
  unsigned getArraySize() const;

  // Function
  std::span<const Type> getParamTypes() const;
  const Type &getReturnType() const;

  // TODO: overload ==
  // TODO: overload !=
  // TODO: hash()
  // TODO: ExtraInfo def (struct/enum)
  // TODO: ExtraInfo array
  // TODO: ExtraInfo function
};

} // namespace dblang::ir
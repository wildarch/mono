#pragma once

#include "ir/StringPool.h"
#include "ir/Type.h"

#include <cstddef>
#include <functional>

namespace dblang::ir::impl {

// types:
// - primitive (bool, char, iX, uX, f32, f64, isize, usize)
// - def (to user-defined struct, enum or alias. can be 'unresolved')
// - array (fixed size)
// - pointer
// - function (for function refs)

enum class TypeKind {
  UNRESOLVED,
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
  // Compound/nested types
  STRUCT, // TODO: maybe remove and have reference to 'Def' instead
  ENUM,   // TODO: maybe remove and have reference to 'Def' instead
  ARRAY,  // fixed-size array
  POINTER,
  FUNCTION, // function reference
};

/// Combine a hash value into a running seed (Fowler-Noll-Vo style).
inline void hashCombine(std::size_t &seed, std::size_t value) {
  // TODO: move to a separate util file
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/// Structural equality and hashing for type implementations.
///
/// Two types are equal iff they have the same kind and, for compound types,
/// the same structure (element types, sizes, field names, etc.). This is
/// independent of the address of the \c TypeImpl, so it can be used to
/// intern/deduplicate types.
struct TypeImpl {
  TypeKind kind;

  bool operator==(const TypeImpl &other) const;
  bool operator!=(const TypeImpl &other) const { return !(*this == other); }
  std::size_t hash() const;
};

// NOTE: no TypeImplPrimitive because TypeImpl::kind already encodes the
// primitive type.

/** Reference to a type known only by its name in the source. */
struct TypeImplUnresolved : public TypeImpl {
  InternedString name;

  bool operator==(const TypeImplUnresolved &other) const {
    return kind == other.kind && name == other.name;
  }
  std::size_t hash() const {
    std::size_t seed = std::hash<TypeKind>{}(kind);
    hashCombine(seed, std::hash<InternedString>{}(name));
    return seed;
  }
};

struct TypeImplStruct : public TypeImpl {
  std::size_t nFields;

  struct Field {
    InternedString name;
    Type type;
  };
  Field fields[];

  static constexpr std::size_t computeAllocationSize(std::size_t nFields) {
    return sizeof(TypeImplStruct) + nFields * sizeof(Field);
  }

  bool operator==(const TypeImplStruct &other) const {
    if (kind != other.kind || nFields != other.nFields)
      return false;
    for (std::size_t i = 0; i < nFields; ++i) {
      if (fields[i].name != other.fields[i].name ||
          fields[i].type != other.fields[i].type)
        return false;
    }
    return true;
  }
  std::size_t hash() const {
    std::size_t seed = std::hash<TypeKind>{}(kind);
    hashCombine(seed, nFields);
    for (std::size_t i = 0; i < nFields; ++i) {
      hashCombine(seed, std::hash<InternedString>{}(fields[i].name));
      hashCombine(seed, fields[i].type.hash());
    }
    return seed;
  }
};

struct TypeImplEnum : public TypeImpl {
  std::size_t nAlts;
  using Alt = InternedString;
  Alt alts[];

  static constexpr std::size_t computeAllocationSize(std::size_t nFields) {
    return sizeof(TypeImplEnum) + nFields * sizeof(Alt);
  }

  bool operator==(const TypeImplEnum &other) const {
    if (kind != other.kind || nAlts != other.nAlts)
      return false;
    for (std::size_t i = 0; i < nAlts; ++i) {
      if (alts[i] != other.alts[i])
        return false;
    }
    return true;
  }
  std::size_t hash() const {
    std::size_t seed = std::hash<TypeKind>{}(kind);
    hashCombine(seed, nAlts);
    for (std::size_t i = 0; i < nAlts; ++i)
      hashCombine(seed, std::hash<InternedString>{}(alts[i]));
    return seed;
  }
};

struct TypeImplArray : public TypeImpl {
  std::size_t size;
  Type elemType;

  bool operator==(const TypeImplArray &other) const {
    return kind == other.kind && size == other.size &&
           elemType == other.elemType;
  }
  std::size_t hash() const {
    std::size_t seed = std::hash<TypeKind>{}(kind);
    hashCombine(seed, size);
    hashCombine(seed, elemType.hash());
    return seed;
  }
};

struct TypeImplPointer : public TypeImpl {
  Type pointee;

  bool operator==(const TypeImplPointer &other) const {
    return kind == other.kind && pointee == other.pointee;
  }
  std::size_t hash() const {
    std::size_t seed = std::hash<TypeKind>{}(kind);
    hashCombine(seed, pointee.hash());
    return seed;
  }
};

struct TypeImplFunction : public TypeImpl {
  std::size_t nParams;
  std::size_t nReturn;
  // nParams + nReturns types (parameters first, return types last)
  Type types[];

  static constexpr std::size_t computeAllocationSize(std::size_t nParams,
                                                     std::size_t nReturn) {
    return sizeof(TypeImplFunction) + (nParams + nReturn) * sizeof(Type);
  }

  bool operator==(const TypeImplFunction &other) const {
    if (kind != other.kind || nParams != other.nParams ||
        nReturn != other.nReturn)
      return false;
    for (std::size_t i = 0; i < nParams + nReturn; ++i) {
      if (types[i] != other.types[i])
        return false;
    }
    return true;
  }
  std::size_t hash() const {
    std::size_t seed = std::hash<TypeKind>{}(kind);
    hashCombine(seed, nParams);
    hashCombine(seed, nReturn);
    for (std::size_t i = 0; i < nParams + nReturn; ++i)
      hashCombine(seed, types[i].hash());
    return seed;
  }
};

} // namespace dblang::ir::impl
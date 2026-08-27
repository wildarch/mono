#include "ir/TypeImpl.h"

namespace dblang::ir::impl {

// Dispatch on the concrete kind. The derived structs are all defined above,
// so we can downcast and delegate to their structural comparison/hash.
bool TypeImpl::operator==(const TypeImpl &other) const {
  if (kind != other.kind)
    return false;
  switch (kind) {
  case TypeKind::UNRESOLVED:
    return static_cast<const TypeImplUnresolved &>(*this) ==
           static_cast<const TypeImplUnresolved &>(other);
  case TypeKind::STRUCT:
    return static_cast<const TypeImplStruct &>(*this) ==
           static_cast<const TypeImplStruct &>(other);
  case TypeKind::ENUM:
    return static_cast<const TypeImplEnum &>(*this) ==
           static_cast<const TypeImplEnum &>(other);
  case TypeKind::ARRAY:
    return static_cast<const TypeImplArray &>(*this) ==
           static_cast<const TypeImplArray &>(other);
  case TypeKind::POINTER:
    return static_cast<const TypeImplPointer &>(*this) ==
           static_cast<const TypeImplPointer &>(other);
  case TypeKind::FUNCTION:
    return static_cast<const TypeImplFunction &>(*this) ==
           static_cast<const TypeImplFunction &>(other);
  // Primitive kinds carry no extra data; equal kinds are equal types.
  default:
    return true;
  }
}

std::size_t TypeImpl::hash() const {
  switch (kind) {
  case TypeKind::UNRESOLVED:
    return static_cast<const TypeImplUnresolved &>(*this).hash();
  case TypeKind::STRUCT:
    return static_cast<const TypeImplStruct &>(*this).hash();
  case TypeKind::ENUM:
    return static_cast<const TypeImplEnum &>(*this).hash();
  case TypeKind::ARRAY:
    return static_cast<const TypeImplArray &>(*this).hash();
  case TypeKind::POINTER:
    return static_cast<const TypeImplPointer &>(*this).hash();
  case TypeKind::FUNCTION:
    return static_cast<const TypeImplFunction &>(*this).hash();
  default:
    return std::hash<TypeKind>{}(kind);
  }
}

} // namespace dblang::ir::impl
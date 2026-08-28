#include "ir/TypePool.h"
#include "ir/StringPool.h"
#include "ir/TypeImpl.h"
#include <alloca.h>
#include <cassert>
#include <cstring>

namespace dblang::ir {

using namespace impl;

Type TypePool::getPrimitive(TypeKind kind) {
  // Primitive types carry no extra data; the kind alone identifies them.
  TypeImpl impl;
  impl.kind = kind;
  return dedupe(sizeof(TypeImpl), alignof(TypeImpl), &impl);
}

Type TypePool::getUnresolved(InternedString name) {
  TypeImplUnresolved impl{TypeKind::UNRESOLVED, name};
  return dedupe(sizeof(TypeImplUnresolved), alignof(TypeImplUnresolved), &impl);
}

Type TypePool::getArray(Type elemType, std::size_t size) {
  TypeImplArray impl;
  impl.kind = TypeKind::ARRAY;
  impl.size = size;
  impl.elemType = elemType;
  return dedupe(sizeof(TypeImplArray), alignof(TypeImplArray), &impl);
}

Type TypePool::getPointer(Type pointee) {
  TypeImplPointer impl;
  impl.kind = TypeKind::POINTER;
  impl.pointee = pointee;
  return dedupe(sizeof(TypeImplPointer), alignof(TypeImplPointer), &impl);
}

Type TypePool::getFunction(std::span<const Type> params,
                           std::span<const Type> returns) {
  const std::size_t nParams = params.size();
  const std::size_t nReturn = returns.size();
  auto size = TypeImplFunction::computeAllocationSize(nParams, nReturn);
  auto *impl = static_cast<TypeImplFunction *>(alloca(size));
  impl->kind = TypeKind::FUNCTION;
  impl->nParams = nParams;
  impl->nReturn = nReturn;
  for (std::size_t i = 0; i < nParams; i++)
    impl->types[i] = params[i];
  for (std::size_t i = 0; i < nReturn; i++)
    impl->types[nParams + i] = returns[i];
  return dedupe(size, alignof(TypeImplFunction), impl);
}

Type TypePool::getDef(const Def *def) {
  TypeImplDef impl;
  impl.kind = TypeKind::DEF;
  impl.def = def;
  return dedupe(sizeof(TypeImplDef), alignof(TypeImplDef), &impl);
}

Type TypePool::dedupe(std::size_t size, std::size_t align,
                      TypeImpl *candidate) {
  auto it = _types.find(candidate);
  if (it != _types.end()) {
    return Type(*it);
  }

  // Copy into the pool. The TypeImpl variants are trivially copyable, so a
  // single memcpy copies the header and any flexible-array members.
  auto *inPool = static_cast<TypeImpl *>(_alloc.allocate(size, align));
  std::memcpy(inPool, candidate, size);
  _types.insert(inPool);
  return Type(inPool);
}

} // namespace dblang::ir

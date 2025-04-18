add_library(
    columnar
    AddSelectionVectors.cpp
    Bufferize.cpp
    Canonicalize.cpp
    Catalog.cpp
    Columnar.cpp
    GroupTableReads.cpp
    LLVMTranslation.cpp
    LowerPipelines.cpp
    LowerToLLVM.cpp
    MakePipelines.cpp
    parquet/ParquetToCatalog.cpp
    PushDownPredicates.cpp
    Runtime.cpp
    runtime/Print.cpp
    runtime/TableColumn.cpp
    runtime/TableScanner.cpp
)

# LLVM/MLIR includes.
# In SYSTEM mode to avoid reporting warnings on these headers.
target_include_directories(columnar
    SYSTEM
    PUBLIC
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

target_include_directories(columnar
    PUBLIC
    ../include
    # Generated MLIR dialect files
    ${CMAKE_BINARY_DIR}/include
)

add_dependencies(
    columnar
    MLIRColumnarIncGen
    MLIRColumnarInterfacesIncGen
    MLIRColumnarPassesIncGen
)

target_link_libraries(
    columnar
    ${llvm_libs}
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRArithTransforms
    MLIRBufferizationTransforms
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRParser
    MLIRPass
    MLIRSCFToControlFlow
    MLIRTensorDialect
    MLIRToLLVMIRTranslationRegistration
    MLIRTransforms
    Parquet::parquet_shared
)

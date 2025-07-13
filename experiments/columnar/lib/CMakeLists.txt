protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS /usr/local/include/pg_query/pg_query.proto)

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
    runtime/PipelineContext.cpp
    runtime/Print.cpp
    runtime/TableColumn.cpp
    runtime/TableScanner.cpp
    runtime/TupleBuffer.cpp
    SQLParser.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS}
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
    # Compiled protobuf .h and .cc are placed here
    ${CMAKE_BINARY_DIR}/lib
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
    ${Protobuf_LIBRARIES}
    libpg_query::libpg_query
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

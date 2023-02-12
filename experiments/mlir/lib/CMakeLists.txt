get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_dialect_library(mini
        MiniDialect.cpp
        MiniOps.cpp
        MiniLoweringPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        MLIRMiniOpsIncGen

	LINK_LIBS PUBLIC
        ${dialect_libs}
        MLIRAnalysis
	MLIRIR
        MLIRInferTypeOpInterface
        MLIRPass
        MLIRSupport
        MLIRTransforms
	)

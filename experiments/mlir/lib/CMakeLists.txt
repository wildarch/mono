add_mlir_dialect_library(mini
        MiniDialect.cpp
        MiniOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        MLIRMiniOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)

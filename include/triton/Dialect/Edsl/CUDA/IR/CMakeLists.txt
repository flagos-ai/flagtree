set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS CUDAOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(CUDAOps CUDAOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS CUDADialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=CUDA)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=CUDA)
add_mlir_doc(CUDADialect CUDADialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS CUDAAttrDefs.td)
mlir_tablegen(CUDAAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(CUDAAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(EdslCUDATableGen)

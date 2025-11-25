# FlagTree Backend Specialization 统一设计（C++）

## 1. 基本设计
FlagTree 为 C++ 代码的后端特化提供的实现方案：使用宏判断在工程编译时选择是否特化。

## 2. td 文件特化
td 文件如果需要特化，可整体复制到对应的后端 spec 目录下进行后端特化实现。例如将 include/triton/Dialect/Triton/IR/TritonAttrDefs.td 复制到 **third_party/iluvatar/backend/spec/**include/triton/Dialect/Triton/IR/TritonAttrDefs.td 进行特化修改（本文以 iluvatar 后端为例），注意不需要修改 td 文件头部的 #ifndef 和 #define 宏，因为 CMakeLists.txt 中通过 set_flagtree_backend_td 方法只选择其中一个进行代码生成。
- include/triton/Dialect/Triton/IR/CMakeLists.txt
```shell
# set(LLVM_TARGET_DEFINITIONS TritonOps.td)  # 原实现
set_flagtree_backend_td(LLVM_TARGET_DEFINITIONS TritonOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(TritonOps TritonOps dialects/ -gen-op-doc)
```

## 3. h 头文件特化

## 4. cpp 文件特化

### 4.1 情形一：整个文件特化
调用关系耦合太多时，可退化为整个文件特化。常用于 cpp 内定义多个 class/struct 并交叉调用的情形。

#### 4.1.1 主干代码的缺省实现
- lib/Dialect/Triton/IR/Ops.cpp
```c++
#if __has_include("flagtree_spec.h")
#include "flagtree_spec.h"
#endif

#ifndef FLAGTREE_SPEC_Dialect_Triton_IR_Ops_cpp
...
#endif
```

#### 4.1.2 宏定义及头文件包含（注意修改文件名及头部宏）
- **third_party/iluvatar/backend/spec/**include/triton/Dialect/Triton/IR/iluvatar_Ops.h
```c++
#ifndef ILUVATAR_TRITON_DIALECT_TRITON_IR_OPS_H_
#define ILUVATAR_TRITON_DIALECT_TRITON_IR_OPS_H_

#define FLAGTREE_SPEC_Dialect_Triton_IR_Ops_cpp

#endif // ILUVATAR_TRITON_DIALECT_TRITON_IR_OPS_H_
```
- third_party/iluvatar/backend/spec/include/flagtree_spec.h
```c++
#include "triton/Dialect/Triton/IR/iluvatar_Ops.h"
```

#### 4.1.3 后端目录的特化实现
- **third_party/iluvatar/backend/spec/**lib/Dialect/Triton/IR/Ops.cpp

### 4.2 特化目标链接
CMakeLists.txt 中通过 get_flagtree_backend_lib 方法将 spec 目录中的特化实现目标链接进来。
- lib/Dialect/Triton/IR/CMakeLists.txt
```shell
get_flagtree_backend_lib("TritonIR" _EXTRA_LINK_LIBS)

add_triton_library(TritonIR
  ....cpp

  DEPENDS
  ...

  LINK_LIBS PUBLIC
  ...
  ${_EXTRA_LINK_LIBS}
)
```




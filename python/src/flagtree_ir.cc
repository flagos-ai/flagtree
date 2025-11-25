#include "ir.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Value.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Edsl/Prelude.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"

namespace py = pybind11;

class FlagTreeOpBuilder : public TritonOpBuilder {};

void init_flagtree_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<flagtree::DSLRegionOp>(m, "DSLRegionOp", py::module_local(),
                                    py::dynamic_attr())
      .def("get_operation", &flagtree::DSLRegionOp::getOperation)
      .def("get_body", &flagtree::DSLRegionOp::getBody, ret::reference)
      .def("dump", &flagtree::DSLRegionOp::dump);

  py::class_<flagtree::YieldOp>(m, "YieldOp", py::module_local(),
                                py::dynamic_attr())
      .def("dump", &flagtree::YieldOp::dump);

  py::class_<FlagTreeOpBuilder, TritonOpBuilder>(
      m, "FlagTreeOpBuilder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("get_op_builder", &FlagTreeOpBuilder::getBuilder, ret::reference)
      .def("create_dsl_region_op",
           [](FlagTreeOpBuilder &self,
              const std::vector<Value> &inputs) -> flagtree::DSLRegionOp {
             return self.create<flagtree::DSLRegionOp>(inputs);
           })
      .def("create_yield_op",
           [](FlagTreeOpBuilder &self) -> flagtree::YieldOp {
             return self.create<flagtree::YieldOp>();
           })
      .def("cuda_create_get_thread_id",
           [](FlagTreeOpBuilder &self, uint32_t axis) -> Value {
             std::optional<edsl::CUDA::ThreadIDDim> dimOpt =
                 edsl::CUDA::symbolizeThreadIDDim(axis);
             if (!dimOpt.has_value()) {
               throw pybind11::index_error("thread_id must be in [0,3]");
             }
             edsl::CUDA::ThreadIDDim dim = *dimOpt;
             return self.create<edsl::CUDA::GetThreadIdOp>(dim);
           });
}

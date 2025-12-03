#include "flagtree/Common/UnifiedHardware.h"

class AscendUnifiedHardware : public mlir::flagtree::UnifiedHardware {
public:
  int getAscendTag() const override;
};

int AscendUnifiedHardware::getAscendTag() const { return 1; }

std::unique_ptr<mlir::flagtree::UnifiedHardware>
mlir::flagtree::createUnifiedHardwareManager() {
  return std::make_unique<AscendUnifiedHardware>();
}

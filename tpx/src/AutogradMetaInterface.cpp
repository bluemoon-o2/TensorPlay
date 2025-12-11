#include "AutogradMetaInterface.h"

namespace tensorplay {
namespace tpx {

static AutogradMetaFactory* global_autograd_meta_factory = nullptr;

void SetAutogradMetaFactory(AutogradMetaFactory* factory) {
    global_autograd_meta_factory = factory;
}

AutogradMetaFactory* GetAutogradMetaFactory() {
    return global_autograd_meta_factory;
}

} // namespace tpx
} // namespace tensorplay

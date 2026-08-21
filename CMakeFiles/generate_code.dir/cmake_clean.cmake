file(REMOVE_RECURSE
  "CMakeFiles/generate_code"
  "build/generated/tensorplay/ops/AutogradNodesGenerated.h"
  "build/generated/tensorplay/ops/TPXAutogradRegistration.cpp"
  "build/generated/tensorplay/ops/TPXOpsGenerated.cpp"
  "build/generated/tensorplay/ops/TPXOpsGenerated.h"
  "build/generated/tensorplay/ops/TensorBindingsGenerated.h"
  "build/generated/tensorplay/ops/TensorGenerated.cpp"
  "build/generated/tensorplay/ops/TensorGenerated.h"
  "build/generated/tensorplay/ops/TensorRedispatchGenerated.h"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/generate_code.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

file(REMOVE_RECURSE
  "libdnnl.a"
  "libdnnl.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang ASM C CXX)
  include(CMakeFiles/dnnl.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

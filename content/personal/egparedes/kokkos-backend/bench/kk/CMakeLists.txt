cmake_minimum_required(VERSION 3.22)
project(gk CXX)
find_package(Kokkos REQUIRED)
add_library(gk SHARED api.cpp reg_cart_it.cpp reg_cart_ex.cpp reg_cart_view.cpp
                      reg_scan.cpp reg_scan_misc.cpp reg_unstr.cpp reg_unstr_misc.cpp)
target_link_libraries(gk PRIVATE Kokkos::kokkos)
# gtfn is built as GNU C++17, where GCC contracts floating point expressions; match it.
target_compile_options(gk PRIVATE -ffp-contract=fast)
if(Kokkos_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(gk PRIVATE CUDA::cudart)
  # per-kernel register usage in the build log
  target_compile_options(gk PRIVATE -Xptxas -v)
endif()

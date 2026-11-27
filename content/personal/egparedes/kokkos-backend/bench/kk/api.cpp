// C interface used by bench.py through ctypes.
#include "gk.hpp"

#include <chrono>
#include <sstream>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace {
// mode 0: the same device synchronisation gtfn's timer uses (cudaDeviceSynchronize);
// mode 1: Kokkos::fence(). On the host both are Kokkos::fence().
void sync(int mode) {
#ifdef KOKKOS_ENABLE_CUDA
  if (mode == 0) {
    cudaDeviceSynchronize();
    return;
  }
#endif
  (void)mode;
  Kokkos::fence();
}
}  // namespace

extern "C" {

void gk_init(int nthreads) {
  Kokkos::InitializationSettings s;
  if (nthreads > 0) s.set_num_threads(nthreads);
  Kokkos::initialize(s);
}

void gk_finalize() {
  gk::pool_clear();
  Kokkos::finalize();
}

void gk_pool_clear() { gk::pool_clear(); }

// Fill every cached temporary with NaN so that a correctness run cannot pass on stale data.
void gk_pool_poison() {
  for (auto &kv : gk::pool_free())
    for (double *p : kv.second) {
      Kokkos::View<double *, gk::Mem, gk::Unmanaged> v(p, kv.first);
      Kokkos::deep_copy(v, __builtin_nan(""));
    }
}

char const *gk_info() {
  static std::string info;
  std::ostringstream o;
  o << "kokkos=" << KOKKOS_VERSION << " space=" << gk::Exec::name()
    << " concurrency=" << gk::Exec().concurrency() << " default_range_index_bytes="
    << sizeof(typename Kokkos::RangePolicy<gk::Exec>::index_type) << " default_md_index_bytes="
    << sizeof(typename Kokkos::MDRangePolicy<gk::Exec, Kokkos::Rank<3>>::index_type);
  info = o.str();
  return info.c_str();
}

char const *gk_variants(char const *kernel) {
  static std::string out;
  out.clear();
  auto it = gk::registry().find(kernel);
  if (it != gk::registry().end())
    for (auto const &kv : it->second) out += (out.empty() ? "" : ",") + kv.first;
  return out.c_str();
}

// Returns the elapsed seconds of one call including the closing synchronisation, < 0 on error.
double gk_run(char const *kernel, char const *variant, GKArgs const *args, int sync_mode) {
  try {
    auto const &fn = gk::registry().at(kernel).at(variant);
    sync(sync_mode);
    auto const t0 = std::chrono::high_resolution_clock::now();
    fn(*args);
    sync(sync_mode);
    auto const t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
  } catch (std::exception const &e) {
    std::fprintf(stderr, "gk_run %s/%s: %s\n", kernel, variant, e.what());
    return -1.0;
  }
}
}

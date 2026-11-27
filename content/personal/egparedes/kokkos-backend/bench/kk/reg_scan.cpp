#include "scan.hpp"
using namespace gk;
using namespace gk::scan;
namespace {
struct R {
  R() {
    using P = Ptr3<int>;
#ifdef KOKKOS_ENABLE_CUDA
    using Base = PerColumn<ColMdTile<int, true>>;
    add<Iter, P, Base>("ptr.it.colmdtlb");
    add<Iter, P, PerColumn<ColMdTile<int, false>>>("ptr.it.colmdt");
    add<Iter, P, PerColumn<ColRange<int, false>>>("ptr.it.colrange");
    add<Iter, P, PerColumn<ColRange<int, true>>>("ptr.it.colrangelb");
    add<Iter, P, PerColumn<ColMdDefault>>("ptr.it.colmddef");
    add<Iter, Ptr3<long>, PerColumn<ColMdTile<long, true>>>("ptr64.it.colmdtlb");
    add_split<Iter, P, Base>("ptr.it.colmdtlb");
    add_tmp<Iter, P, Base>("ptr.it.colmdtlb");
    add_layouts<Iter, Base, false>("ptr.it.colmdtlb");
    add_layouts<Iter, Base, true>("ptr.it.colmdtlb");
    add_layouts<Iter, PerColumn<ColRange<int, false>>, false>("ptr.it.colrange");
#else
    add<Iter, P, Chunk<8>>("ptr.it.chunk8");
    add<Iter, P, Chunk<16>>("ptr.it.chunk16");
    add<Iter, P, Chunk<64>>("ptr.it.chunk64");
    add<Iter, P, PerColumn<ColRange<int>>>("ptr.it.col");
    add<Iter, Ptr3<long>, PerColumn<ColRange<long>>>("ptr64.it.col");
    add<Iter, Ptr3<long>, Chunk<16>>("ptr64.it.chunk16");
    add_split<Iter, P, PerColumn<ColRange<int>>>("ptr.it.col");
    add_split<Iter, P, Chunk<16>>("ptr.it.chunk16");
    add_tmp<Iter, P, Chunk<16>>("ptr.it.chunk16");
    add_tmp<Iter, P, PerColumn<ColRange<int>>>("ptr.it.col");
    // host registration only serves the correctness check of the layout code
    add_layouts<Iter, PerColumn<ColRange<int>>, false>("ptr.it.col");
    add_layouts<Iter, PerColumn<ColRange<int>>, true>("ptr.it.col");
#endif
  }
} r;
}  // namespace

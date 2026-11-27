#include "scan.hpp"
using namespace gk;
using namespace gk::scan;
namespace {
struct R {
  R() {
    using P = Ptr3<int>;
#ifdef KOKKOS_ENABLE_CUDA
    using Base = PerColumn<ColMdTile<int, true>>;
    add<Expl, P, Base>("ptr.ex.colmdtlb");
    add<Expl, P, PerColumn<ColMdTile<int, false>>>("ptr.ex.colmdt");
    add<Iter, ViewL3, Base>("vl.it.colmdtlb");
    add<Expl, ViewL3, Base>("vl.ex.colmdtlb");
    add<Iter, ViewS3, Base>("vs.it.colmdtlb");
#else
    add<Expl, P, Chunk<16>>("ptr.ex.chunk16");
    add<Expl, P, Chunk<8>>("ptr.ex.chunk8");
    add<Expl, P, Chunk<32>>("ptr.ex.chunk32");
    add<Expl, P, Chunk<64>>("ptr.ex.chunk64");
    add<Expl, P, PerColumn<ColRange<int>>>("ptr.ex.col");
    add<Iter, ViewL3, PerColumn<ColRange<int>>>("vl.it.col");
    add<Iter, ViewL3, Chunk<16>>("vl.it.chunk16");
    add<Expl, ViewL3, Chunk<16>>("vl.ex.chunk16");
    add<Iter, ViewS3, Chunk<16>>("vs.it.chunk16");
#endif
    add<Iter, ViewS3, PerColumn<ColMdDefault>>("naive");
#ifdef KOKKOS_ENABLE_CUDA
    add<Walk, P, Base>("ptr.pw.colmdtlb");
    add<Walk, P, PerColumn<ColRange<int, false>>>("ptr.pw.colrange");
    add_split<Walk, P, Base>("ptr.pw.colmdtlb");
    add_split_tmp<Walk, P, Base>("ptr.pw.colmdtlb");
    add_split_tmp<Iter, P, Base>("ptr.it.colmdtlb");
#else
    add<Walk, P, PerColumn<ColRange<int>>>("ptr.pw.col");
#endif
  }
} r;
}  // namespace

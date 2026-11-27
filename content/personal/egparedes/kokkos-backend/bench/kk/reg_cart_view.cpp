#include "cart.hpp"
using namespace gk;
using namespace gk::cart;
namespace {
struct R {
  R() {
#ifdef KOKKOS_ENABLE_CUDA
    add<Iter, ViewL3, MdTile<int, true>>("vl.it.mdtlb");
    add<Expl, ViewL3, MdTile<int, true>>("vl.ex.mdtlb");
    add<Iter, ViewS3, MdTile<int, true>>("vs.it.mdtlb");
#else
    add<Iter, ViewL3, Rows<int>>("vl.it.rows");
    add<Expl, ViewL3, Rows<int>>("vl.ex.rows");
    add<Iter, ViewS3, Rows<int>>("vs.it.rows");
    add<Iter, ViewL3, MdLeft<int>>("vl.it.mdleft");
#endif
    // negative control: Kokkos defaults everywhere
    add<Iter, ViewS3, MdDefault>("naive");
#ifdef KOKKOS_ENABLE_CUDA
    add<Expl, Ptr3<int>, MdDefault>("ptr.ex.mddef");
#else
    add<Iter, Ptr3<int>, MdDefault>("ptr.it.mddef");
    add<Iter, Ptr3<int>, RowsSimd<int>>("ptr.it.rowssimd");
#endif
  }
} r;
}  // namespace

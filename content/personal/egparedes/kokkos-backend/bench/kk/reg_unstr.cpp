#include "unstr.hpp"
using namespace gk;
using namespace gk::unstr;
namespace {
struct R {
  R() {
    using P = Ptr2<int>;
    using T = TabPtr<int>;
#ifdef KOKKOS_ENABLE_CUDA
    add<Iter, P, T, HMdTile<int, true>>("ptr.it.mdtlb");
    add<Iter, P, T, HMdTile<int, false>>("ptr.it.mdt");
    add<Iter, P, T, HMdDefault>("ptr.it.mddef");
    add<Iter, P, T, HMdDefaultInt>("ptr.it.mddefint");
    add<Iter, P, T, HRange>("ptr.it.range");
    add<Iter, Ptr2<long>, TabPtr<long>, HMdTile<long, true>>("ptr64.it.mdtlb");
#else
    add<Iter, P, T, HRows<int>>("ptr.it.rows");
    add<Iter, Ptr2<long>, TabPtr<long>, HRows<long>>("ptr64.it.rows");
    add<Iter, P, T, HRows<int, false>>("ptr.it.rowsnh");
    add<Iter, P, T, HMdLeft<int>>("ptr.it.mdleft");
#endif
  }
} r;
}  // namespace

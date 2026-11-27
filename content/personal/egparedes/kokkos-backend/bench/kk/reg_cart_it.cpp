#include "cart.hpp"
using namespace gk;
using namespace gk::cart;
namespace {
struct R {
  R() {
    using P = Ptr3<int>;
    using P64 = Ptr3<long>;
#ifdef KOKKOS_ENABLE_CUDA
    add<Iter, P, MdTile<int, true>>("ptr.it.mdtlb");
    add<Iter, P, MdTile<int, false>>("ptr.it.mdt");
    add<Iter, P, MdDefault>("ptr.it.mddef");
    add<Iter, P, MdDefaultInt>("ptr.it.mddefint");
    add<Iter, P, Range3>("ptr.it.range");
    add<Iter, P64, MdTile<long, true>>("ptr64.it.mdtlb");
    add_tmp<Iter, P, MdTile<int, true>>("ptr.it.mdtlb");
#else
    add<Iter, P, Rows<int>>("ptr.it.rows");
    add<Iter, P64, Rows<long>>("ptr64.it.rows");
    add<Iter, P, Rows<int, false>>("ptr.it.rowsnh");
    add<Iter, P, MdLeft<int>>("ptr.it.mdleft");
    add<Iter, P64, MdLeft<long>>("ptr64.it.mdleft");
    add_tmp<Iter, P, Rows<int>>("ptr.it.rows");
#endif
  }
} r;
}  // namespace

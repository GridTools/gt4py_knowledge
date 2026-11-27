#include "cart.hpp"
using namespace gk;
using namespace gk::cart;
namespace {
struct R {
  R() {
    using P = Ptr3<int>;
#ifdef KOKKOS_ENABLE_CUDA
    add<Expl, P, MdTile<int, true>>("ptr.ex.mdtlb");
    add<Expl, P, MdTile<int, false>>("ptr.ex.mdt");
#else
    add<Expl, P, Rows<int>>("ptr.ex.rows");
    add<Expl, P, MdLeft<int>>("ptr.ex.mdleft");
#endif
  }
} r;
}  // namespace

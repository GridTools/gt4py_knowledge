#include "unstr.hpp"
using namespace gk;
using namespace gk::unstr;
namespace {
struct R {
  R() {
    using P = Ptr2<int>;
    using T = TabPtr<int>;
#ifdef KOKKOS_ENABLE_CUDA
    add<Expl, P, T, HMdTile<int, true>>("ptr.ex.mdtlb");
    add<Expl, P, T, HMdTile<int, false>>("ptr.ex.mdt");
    add<Iter, ViewL2, TabViewL, HMdTile<int, true>>("vl.it.mdtlb");
    add<Expl, ViewL2, TabViewL, HMdTile<int, true>>("vl.ex.mdtlb");
    add<Iter, ViewS2, TabViewS, HMdTile<int, true>>("vs.it.mdtlb");
#else
    add<Expl, P, T, HRows<int>>("ptr.ex.rows");
    add<Expl, P, T, HRows<int, false>>("ptr.ex.rowsnh");
    add<Expl, P, T, HMdLeft<int>>("ptr.ex.mdleft");
    add<Iter, ViewL2, TabViewL, HRows<int>>("vl.it.rows");
    add<Expl, ViewL2, TabViewL, HRows<int>>("vl.ex.rows");
    add<Iter, ViewS2, TabViewS, HRows<int>>("vs.it.rows");
#endif
    add<Iter, ViewS2, TabViewS, HMdDefault>("naive");
#ifdef KOKKOS_ENABLE_CUDA
    using L = HMdTile<int, true>;
    reg("v2e", "ptr.it1.mdtlb", &run_v2e<Iter1, P, T, L>);
    reg("v2e", "ptr.it1.mddef", &run_v2e<Iter1, P, T, HMdDefault>);
    reg("v2e", "ptr.ex2.mdtlb", &run_v2e<Expl2, P, T, L>);
    reg("chain", "ptr.ex2.mdtlb", &run_chain<Expl2, P, T, L>);
    reg("chain", "ptr.itlit.mdtlb", &run_chain<IterLit, P, T, L>);
    add<Expl, P, T, HMdDefault>("ptr.ex.mddef");
#else
    using L = HRows<int>;
    reg("v2e", "ptr.it1.rows", &run_v2e<Iter1, P, T, L>);
    reg("v2e", "ptr.ex2.rows", &run_v2e<Expl2, P, T, L>);
    reg("chain", "ptr.ex2.rows", &run_chain<Expl2, P, T, L>);
    reg("chain", "ptr.itlit.rows", &run_chain<IterLit, P, T, L>);
    add<Iter, P, T, HMdDefault>("ptr.it.mddef");
#endif
  }
} r;
}  // namespace

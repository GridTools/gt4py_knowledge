// Small "gk" runtime used by the hand-written Kokkos kernels: field accessors, launch shapes,
// temporary storage and a variant registry. The same sources build for OpenMP and CUDA.
#pragma once

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define GK_INLINE KOKKOS_FORCEINLINE_FUNCTION

// Mirrored (by field order and type) in bench.py.
struct GKArgs {
  double *f[8];
  std::int64_t fs[8][3];  // element strides
  std::int64_t fe[8][3];  // extents
  const std::int32_t *t[2];
  std::int64_t ts[2][2];
  std::int64_t te[2][2];
  std::int32_t ni, nj, nk, h;  // unstructured: ni = number of output horizontal points
};

namespace gk {

using Exec = Kokkos::DefaultExecutionSpace;
using Mem = Exec::memory_space;
using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

// ---------------------------------------------------------------- registry
using Fn = std::function<void(GKArgs const &)>;
inline std::map<std::string, std::map<std::string, Fn>> &registry() {
  static std::map<std::string, std::map<std::string, Fn>> r;
  return r;
}
inline void reg(std::string const &kernel, std::string const &variant, Fn fn) {
  if (!registry()[kernel].emplace(variant, std::move(fn)).second)
    std::fprintf(stderr, "gk: duplicate variant %s/%s\n", kernel.c_str(), variant.c_str());
}

inline void require(bool ok, char const *what) {
  if (!ok) throw std::runtime_error(what);
}

// ---------------------------------------------------------------- 3-d field accessors (i, j, k)
template <class Idx>
struct Ptr3 {  // pointer + strides, unit stride in i known at compile time
  using index_type = Idx;
  double *p;
  Idx sj, sk;
  static Ptr3 wrap(double *p, GKArgs const &a, int like) {
    require(a.fs[like][0] == 1, "Ptr3: i stride is not 1");
    return {p, static_cast<Idx>(a.fs[like][1]), static_cast<Idx>(a.fs[like][2])};
  }
  GK_INLINE double &operator()(Idx i, Idx j, Idx k) const { return p[i + j * sj + k * sk]; }
};

struct ViewL3 {  // unmanaged LayoutLeft View over the padded extents
  using index_type = int;
  using V = Kokkos::View<double ***, Kokkos::LayoutLeft, Mem, Unmanaged>;
  V v;
  static ViewL3 wrap(double *p, GKArgs const &a, int like) {
    require(a.fs[like][0] == 1 && a.fs[like][2] % a.fs[like][1] == 0, "ViewL3: not padded-left");
    return {V(p, a.fs[like][1], a.fs[like][2] / a.fs[like][1], a.fe[like][2])};
  }
  GK_INLINE double &operator()(int i, int j, int k) const { return v(i, j, k); }
};

struct ViewS3 {  // unmanaged LayoutStride View
  using index_type = int;
  using V = Kokkos::View<double ***, Kokkos::LayoutStride, Mem, Unmanaged>;
  V v;
  static ViewS3 wrap(double *p, GKArgs const &a, int like) {
    Kokkos::LayoutStride l(a.fe[like][0], a.fs[like][0], a.fe[like][1], a.fs[like][1],
                           a.fe[like][2], a.fs[like][2]);
    return {V(p, l)};
  }
  GK_INLINE double &operator()(int i, int j, int k) const { return v(i, j, k); }
};

template <class A>
A field(GKArgs const &a, int f) {
  return A::wrap(a.f[f], a, f);
}

// ---------------------------------------------------------------- 2-d field accessors (h, k)
template <class Idx>
struct Ptr2 {
  using index_type = Idx;
  double *p;
  Idx sk;
  static Ptr2 wrap(double *p, GKArgs const &a, int like) {
    require(a.fs[like][0] == 1, "Ptr2: horizontal stride is not 1");
    return {p, static_cast<Idx>(a.fs[like][1])};
  }
  GK_INLINE double &operator()(Idx h, Idx k) const { return p[h + k * sk]; }
};

struct ViewL2 {
  using index_type = int;
  using V = Kokkos::View<double **, Kokkos::LayoutLeft, Mem, Unmanaged>;
  V v;
  static ViewL2 wrap(double *p, GKArgs const &a, int like) {
    require(a.fs[like][0] == 1, "ViewL2: horizontal stride is not 1");
    return {V(p, a.fs[like][1], a.fe[like][1])};
  }
  GK_INLINE double &operator()(int h, int k) const { return v(h, k); }
};

struct ViewS2 {
  using index_type = int;
  using V = Kokkos::View<double **, Kokkos::LayoutStride, Mem, Unmanaged>;
  V v;
  static ViewS2 wrap(double *p, GKArgs const &a, int like) {
    Kokkos::LayoutStride l(a.fe[like][0], a.fs[like][0], a.fe[like][1], a.fs[like][1]);
    return {V(p, l)};
  }
  GK_INLINE double &operator()(int h, int k) const { return v(h, k); }
};

// ---------------------------------------------------------------- connectivity tables (h, n)
template <class Idx>
struct TabPtr {
  std::int32_t const *p;
  Idx sn;
  static TabPtr make(GKArgs const &a, int t) {
    require(a.ts[t][0] == 1, "TabPtr: horizontal stride is not 1");
    return {a.t[t], static_cast<Idx>(a.ts[t][1])};
  }
  GK_INLINE std::int32_t operator()(Idx h, int n) const { return p[h + n * sn]; }
};

struct TabViewL {
  using V = Kokkos::View<std::int32_t const **, Kokkos::LayoutLeft, Mem, Unmanaged>;
  V v;
  static TabViewL make(GKArgs const &a, int t) {
    require(a.ts[t][0] == 1, "TabViewL: horizontal stride is not 1");
    return {V(a.t[t], a.ts[t][1], a.te[t][1])};
  }
  GK_INLINE std::int32_t operator()(int h, int n) const { return v(h, n); }
};

struct TabViewS {
  using V = Kokkos::View<std::int32_t const **, Kokkos::LayoutStride, Mem, Unmanaged>;
  V v;
  static TabViewS make(GKArgs const &a, int t) {
    Kokkos::LayoutStride l(a.te[t][0], a.ts[t][0], a.te[t][1], a.ts[t][1]);
    return {V(a.t[t], l)};
  }
  GK_INLINE std::int32_t operator()(int h, int n) const { return v(h, n); }
};

// ---------------------------------------------------------------- temporaries
inline std::map<std::size_t, std::vector<double *>> &pool_free() {
  static std::map<std::size_t, std::vector<double *>> p;
  return p;
}
inline void pool_clear() {
  for (auto &kv : pool_free())
    for (double *p : kv.second) Kokkos::kokkos_free<Mem>(p);
  pool_free().clear();
}

struct TmpPool {  // size-keyed cache, nothing is initialised
  double *p;
  std::size_t n;
  explicit TmpPool(std::size_t n) : n(n) {
    auto &s = pool_free()[n];
    if (s.empty()) {
      p = static_cast<double *>(Kokkos::kokkos_malloc<Mem>("gk_pool", n * sizeof(double)));
    } else {
      p = s.back();
      s.pop_back();
    }
  }
  TmpPool(TmpPool const &) = delete;
  ~TmpPool() { pool_free()[n].push_back(p); }
  double *data() const { return p; }
};

struct TmpAlloc {  // kokkos_malloc / kokkos_free on every call
  double *p;
  explicit TmpAlloc(std::size_t n)
      : p(static_cast<double *>(Kokkos::kokkos_malloc<Mem>("gk_tmp", n * sizeof(double)))) {}
  TmpAlloc(TmpAlloc const &) = delete;
  ~TmpAlloc() { Kokkos::kokkos_free<Mem>(p); }
  double *data() const { return p; }
};

struct TmpView {  // default managed View on every call (zero-initialised)
  Kokkos::View<double *, Mem> v;
  explicit TmpView(std::size_t n) : v("gk_view", n) {}
  double *data() const { return v.data(); }
};

struct TmpViewNoInit {
  Kokkos::View<double *, Mem> v;
  explicit TmpViewNoInit(std::size_t n)
      : v(Kokkos::view_alloc(std::string("gk_view"), Kokkos::WithoutInitializing), n) {}
  double *data() const { return v.data(); }
};

// ---------------------------------------------------------------- launch shapes, 3-d stencils
struct Box3 {
  int i0, i1, j0, j1, k0, k1;
};

#ifdef KOKKOS_ENABLE_CUDA
using Bounds256 = Kokkos::LaunchBounds<256>;
#endif

// Kokkos defaults for everything: index type, tile sizes, iteration order.
struct MdDefault {
  template <class B>
  static void run(Box3 d, B const &body) {
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>> p({d.i0, d.j0, d.k0}, {d.i1, d.j1, d.k1});
    Kokkos::parallel_for("gk_md_default", p, body);
  }
};

struct MdDefaultInt {  // default tiles and order, 32-bit index
  template <class B>
  static void run(Box3 d, B const &body) {
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<3>, Kokkos::IndexType<int>> p({d.i0, d.j0, d.k0},
                                                                           {d.i1, d.j1, d.k1});
    Kokkos::parallel_for("gk_md_default_int", p, body);
  }
};

#ifdef KOKKOS_ENABLE_CUDA
template <class Idx, bool LB>
struct MdTile {  // tile {32, 8, 1}, i fastest, optional launch bounds
  template <class B>
  static void run(Box3 d, B const &body) {
    using R = Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>;
    if constexpr (LB) {
      Kokkos::MDRangePolicy<Exec, R, Kokkos::IndexType<Idx>, Bounds256> p(
          {d.i0, d.j0, d.k0}, {d.i1, d.j1, d.k1}, {32, 8, 1});
      Kokkos::parallel_for("gk_md_tile_lb", p, body);
    } else {
      Kokkos::MDRangePolicy<Exec, R, Kokkos::IndexType<Idx>> p({d.i0, d.j0, d.k0},
                                                               {d.i1, d.j1, d.k1}, {32, 8, 1});
      Kokkos::parallel_for("gk_md_tile", p, body);
    }
  }
};

struct Range3 {  // flat RangePolicy, one thread per point, index decoded by division
  template <class B>
  struct F {
    B body;
    int i0, j0, k0, ni, nij;
    GK_INLINE void operator()(int w) const {
      int const k = w / nij, r = w - k * nij, j = r / ni, i = r - j * ni;
      body(i0 + i, j0 + j, k0 + k);
    }
  };
  template <class B>
  static void run(Box3 d, B const &body) {
    int const ni = d.i1 - d.i0, nij = ni * (d.j1 - d.j0);
    Kokkos::parallel_for("gk_range", Kokkos::RangePolicy<Exec, Kokkos::IndexType<int>>(
                                         0, nij * (d.k1 - d.k0)),
                         F<B>{body, d.i0, d.j0, d.k0, ni, nij});
  }
};
#endif

template <class Idx>
struct MdLeft {  // per-point functor, MDRange with i fastest
  template <class B>
  static void run(Box3 d, B const &body) {
    using R = Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>;
    Kokkos::MDRangePolicy<Exec, R, Kokkos::IndexType<Idx>> p({d.i0, d.j0, d.k0},
                                                             {d.i1, d.j1, d.k1});
    Kokkos::parallel_for("gk_md_left", p, body);
  }
};

// Header-owned host loop: RangePolicy over the (k, j) rows, inner unit-stride i loop.
// Hoist = true copies the functor into locals before the inner loop.
template <class Idx, bool Hoist = true>
struct Rows {
  template <class B>
  struct F {
    B body;
    Idx i0, i1, j0, nj, k0;
    GK_INLINE void operator()(Idx w) const {
      Idx const k = k0 + w / nj, j = j0 + w % nj;
      if constexpr (Hoist) {
        B const b = body;
        Idx const lo = i0, hi = i1;
        for (Idx i = lo; i < hi; ++i) b(i, j, k);
      } else {
        for (Idx i = i0; i < i1; ++i) body(i, j, k);
      }
    }
  };
  template <class B>
  static void run(Box3 d, B const &body) {
    Idx const nj = d.j1 - d.j0;
    Kokkos::parallel_for(
        "gk_rows",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<Idx>>(0, nj * static_cast<Idx>(d.k1 - d.k0)),
        F<B>{body, static_cast<Idx>(d.i0), static_cast<Idx>(d.i1), static_cast<Idx>(d.j0), nj,
             static_cast<Idx>(d.k0)});
  }
};

#ifndef KOKKOS_ENABLE_CUDA
// Rows with an explicit simd directive on the inner loop.
template <class Idx>
struct RowsSimd {
  template <class B>
  struct F {
    B body;
    Idx i0, i1, j0, nj, k0;
    void operator()(Idx w) const {
      Idx const k = k0 + w / nj, j = j0 + w % nj;
      B const b = body;
      Idx const lo = i0, hi = i1;
#pragma omp simd
      for (Idx i = lo; i < hi; ++i) b(i, j, k);
    }
  };
  template <class B>
  static void run(Box3 d, B const &body) {
    Idx const nj = d.j1 - d.j0;
    Kokkos::parallel_for(
        "gk_rows_simd",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<Idx>>(0, nj * static_cast<Idx>(d.k1 - d.k0)),
        F<B>{body, static_cast<Idx>(d.i0), static_cast<Idx>(d.i1), static_cast<Idx>(d.j0), nj,
             static_cast<Idx>(d.k0)});
  }
};
#endif

// ---------------------------------------------------------------- launch shapes, 2-d (h, k)
struct HMdDefault {
  template <class B>
  static void run(int nh, int nk, B const &body) {
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>> p({0, 0}, {nh, nk});
    Kokkos::parallel_for("gk_h_md_default", p, body);
  }
};
struct HMdDefaultInt {
  template <class B>
  static void run(int nh, int nk, B const &body) {
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>, Kokkos::IndexType<int>> p({0, 0}, {nh, nk});
    Kokkos::parallel_for("gk_h_md_default_int", p, body);
  }
};
#ifdef KOKKOS_ENABLE_CUDA
template <class Idx, bool LB>
struct HMdTile {  // tile {32, 8}: horizontal x vertical, as gtfn_gpu's thread blocks
  template <class B>
  static void run(int nh, int nk, B const &body) {
    using R = Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>;
    if constexpr (LB) {
      Kokkos::MDRangePolicy<Exec, R, Kokkos::IndexType<Idx>, Bounds256> p({0, 0}, {nh, nk},
                                                                          {32, 8});
      Kokkos::parallel_for("gk_h_md_tile_lb", p, body);
    } else {
      Kokkos::MDRangePolicy<Exec, R, Kokkos::IndexType<Idx>> p({0, 0}, {nh, nk}, {32, 8});
      Kokkos::parallel_for("gk_h_md_tile", p, body);
    }
  }
};
struct HRange {
  template <class B>
  struct F {
    B body;
    int nh;
    GK_INLINE void operator()(int w) const { body(w % nh, w / nh); }
  };
  template <class B>
  static void run(int nh, int nk, B const &body) {
    Kokkos::parallel_for("gk_h_range", Kokkos::RangePolicy<Exec, Kokkos::IndexType<int>>(0, nh * nk),
                         F<B>{body, nh});
  }
};
#endif
template <class Idx>
struct HMdLeft {
  template <class B>
  static void run(int nh, int nk, B const &body) {
    using R = Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>;
    Kokkos::MDRangePolicy<Exec, R, Kokkos::IndexType<Idx>> p({0, 0}, {nh, nk});
    Kokkos::parallel_for("gk_h_md_left", p, body);
  }
};
// Outer range over (k, horizontal block of 1024), inner unit-stride horizontal loop.
template <class Idx, bool Hoist = true>
struct HRows {
  static constexpr int block = 1024;
  template <class B>
  struct F {
    B body;
    Idx nh, nblk;
    GK_INLINE void operator()(Idx w) const {
      Idx const k = w / nblk, h0 = (w % nblk) * block;
      Idx const h1 = h0 + block < nh ? h0 + block : nh;
      if constexpr (Hoist) {
        B const b = body;
        for (Idx h = h0; h < h1; ++h) b(h, k);
      } else {
        for (Idx h = h0; h < h1; ++h) body(h, k);
      }
    }
  };
  template <class B>
  static void run(int nh, int nk, B const &body) {
    Idx const nblk = (nh + block - 1) / block;
    Kokkos::parallel_for(
        "gk_h_rows",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<Idx>>(0, nblk * static_cast<Idx>(nk)),
        F<B>{body, static_cast<Idx>(nh), nblk});
  }
};

// ---------------------------------------------------------------- launch shapes, columns (i, j)
struct Box2 {
  int i0, i1, j0, j1;
};

struct ColMdDefault {
  template <class B>
  static void run(Box2 d, B const &body) {
    Kokkos::MDRangePolicy<Exec, Kokkos::Rank<2>> p({d.i0, d.j0}, {d.i1, d.j1});
    Kokkos::parallel_for("gk_col_md_default", p, body);
  }
};
#ifdef KOKKOS_ENABLE_CUDA
template <class Idx, bool LB>
struct ColMdTile {  // one thread per column, tile {32, 8}
  template <class B>
  static void run(Box2 d, B const &body) {
    using R = Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>;
    if constexpr (LB) {
      Kokkos::MDRangePolicy<Exec, R, Kokkos::IndexType<Idx>, Bounds256> p({d.i0, d.j0},
                                                                          {d.i1, d.j1}, {32, 8});
      Kokkos::parallel_for("gk_col_md_tile_lb", p, body);
    } else {
      Kokkos::MDRangePolicy<Exec, R, Kokkos::IndexType<Idx>> p({d.i0, d.j0}, {d.i1, d.j1},
                                                               {32, 8});
      Kokkos::parallel_for("gk_col_md_tile", p, body);
    }
  }
};
#endif
template <class Idx, bool LB = false>
struct ColRange {  // flat range, one column per iteration (GPU: one thread per column)
  template <class B>
  struct F {
    B body;
    Idx i0, j0, ni;
    GK_INLINE void operator()(Idx w) const { body(i0 + w % ni, j0 + w / ni); }
  };
  template <class B>
  static void run(Box2 d, B const &body) {
    Idx const ni = d.i1 - d.i0, n = ni * (d.j1 - d.j0);
    F<B> f{body, static_cast<Idx>(d.i0), static_cast<Idx>(d.j0), ni};
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (LB) {
      Kokkos::parallel_for("gk_col_range_lb",
                           Kokkos::RangePolicy<Exec, Kokkos::IndexType<Idx>, Bounds256>(0, n), f);
    } else
#endif
    {
      Kokkos::parallel_for("gk_col_range",
                           Kokkos::RangePolicy<Exec, Kokkos::IndexType<Idx>>(0, n), f);
    }
  }
};

}  // namespace gk

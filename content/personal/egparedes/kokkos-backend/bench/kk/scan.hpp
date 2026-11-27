// Tridiagonal solve: forward sweep with tuple state (cp, dp), backward sweep. "merged" runs both
// sweeps in one column kernel; "split" uses one kernel per sweep, as gtfn does.
#pragma once

#include "cart.hpp"

#include <set>

namespace gk::scan {

struct State {
  double cp, dp;
};

struct Iter {
  template <class It>
  GK_INLINE static State fwd(State s, It const &a, It const &b, It const &c, It const &d) {
    double const av = a.deref();
    double const den = b.deref() - av * s.cp;
    return {c.deref() / den, (d.deref() - av * s.dp) / den};
  }
  template <class It>
  GK_INLINE static double bwd(double x, It const &cp, It const &dp) {
    return dp.deref() - cp.deref() * x;
  }
};

// One column (i, j); iterators are advanced along k, as a column executor does.
template <class S, class A>
struct Tri;

template <class A>
struct Tri<Iter, A> {
  using I = typename A::index_type;
  using It = gk::cart::It3<A>;
  A a, b, c, d, cp, dp, x;
  I nk;
  GK_INLINE void forward(I i, I j) const {
    State s{0.0, 0.0};
    It ai{a, i, j, 0}, bi{b, i, j, 0}, ci{c, i, j, 0}, di{d, i, j, 0};
    It cpo{cp, i, j, 0}, dpo{dp, i, j, 0};
    for (I k = 0; k < nk; ++k) {
      s = Iter::fwd(s, ai, bi, ci, di);
      cpo.a(cpo.i, cpo.j, cpo.k) = s.cp;
      dpo.a(dpo.i, dpo.j, dpo.k) = s.dp;
      ++ai.k, ++bi.k, ++ci.k, ++di.k, ++cpo.k, ++dpo.k;
    }
  }
  GK_INLINE void backward(I i, I j) const {
    double xv = 0.0;
    It cpi{cp, i, j, nk - 1}, dpi{dp, i, j, nk - 1}, xo{x, i, j, nk - 1};
    for (I k = nk - 1; k >= 0; --k) {
      xv = Iter::bwd(xv, cpi, dpi);
      xo.a(xo.i, xo.j, xo.k) = xv;
      --cpi.k, --dpi.k, --xo.k;
    }
  }
  // B consecutive columns: k outer, lanes inner (host vectorisation)
  template <int B>
  GK_INLINE void forward_chunk(I i0, I j, int n) const {
    double scp[B], sdp[B];
    for (int l = 0; l < n; ++l) scp[l] = 0.0, sdp[l] = 0.0;
    for (I k = 0; k < nk; ++k) {
#ifndef KOKKOS_ENABLE_CUDA
#pragma omp simd
#endif
      for (int l = 0; l < n; ++l) {
        I const i = i0 + l;
        State const s = Iter::fwd(State{scp[l], sdp[l]}, It{a, i, j, k}, It{b, i, j, k},
                                  It{c, i, j, k}, It{d, i, j, k});
        scp[l] = s.cp, sdp[l] = s.dp;
        cp(i, j, k) = s.cp;
        dp(i, j, k) = s.dp;
      }
    }
  }
  template <int B>
  GK_INLINE void backward_chunk(I i0, I j, int n) const {
    double xv[B];
    for (int l = 0; l < n; ++l) xv[l] = 0.0;
    for (I k = nk - 1; k >= 0; --k) {
#ifndef KOKKOS_ENABLE_CUDA
#pragma omp simd
#endif
      for (int l = 0; l < n; ++l) {
        I const i = i0 + l;
        xv[l] = Iter::bwd(xv[l], It{cp, i, j, k}, It{dp, i, j, k});
        x(i, j, k) = xv[l];
      }
    }
  }
};

struct Expl {};

template <class A>
struct Tri<Expl, A> {
  using I = typename A::index_type;
  A a, b, c, d, cp, dp, x;
  I nk;
  GK_INLINE void forward(I i, I j) const {
    double scp = 0.0, sdp = 0.0;
    for (I k = 0; k < nk; ++k) {
      double const av = a(i, j, k);
      double const den = b(i, j, k) - av * scp;
      scp = c(i, j, k) / den;
      sdp = (d(i, j, k) - av * sdp) / den;
      cp(i, j, k) = scp;
      dp(i, j, k) = sdp;
    }
  }
  GK_INLINE void backward(I i, I j) const {
    double xv = 0.0;
    for (I k = nk - 1; k >= 0; --k) {
      xv = dp(i, j, k) - cp(i, j, k) * xv;
      x(i, j, k) = xv;
    }
  }
  template <int B>
  GK_INLINE void forward_chunk(I i0, I j, int n) const {
    double scp[B], sdp[B];
    for (int l = 0; l < n; ++l) scp[l] = 0.0, sdp[l] = 0.0;
    for (I k = 0; k < nk; ++k) {
#ifndef KOKKOS_ENABLE_CUDA
#pragma omp simd
#endif
      for (int l = 0; l < n; ++l) {
        I const i = i0 + l;
        double const av = a(i, j, k);
        double const den = b(i, j, k) - av * scp[l];
        scp[l] = c(i, j, k) / den;
        sdp[l] = (d(i, j, k) - av * sdp[l]) / den;
        cp(i, j, k) = scp[l];
        dp(i, j, k) = sdp[l];
      }
    }
  }
  template <int B>
  GK_INLINE void backward_chunk(I i0, I j, int n) const {
    double xv[B];
    for (int l = 0; l < n; ++l) xv[l] = 0.0;
    for (I k = nk - 1; k >= 0; --k) {
#ifndef KOKKOS_ENABLE_CUDA
#pragma omp simd
#endif
      for (int l = 0; l < n; ++l) {
        I const i = i0 + l;
        xv[l] = dp(i, j, k) - cp(i, j, k) * xv[l];
        x(i, j, k) = xv[l];
      }
    }
  }
};

// Pointer-walking column: one base pointer per field and column, advanced by the k stride,
// as the gtfn column executor does. Host and device, per-column launch shapes only.
struct Walk {};

template <class A>
struct Tri<Walk, A> {
  using I = typename A::index_type;
  A a, b, c, d, cp, dp, x;
  I nk;
  GK_INLINE void forward(I i, I j) const {
    double const *pa = &a(i, j, 0), *pb = &b(i, j, 0), *pc = &c(i, j, 0), *pd = &d(i, j, 0);
    double *pcp = &cp(i, j, 0), *pdp = &dp(i, j, 0);
    double scp = 0.0, sdp = 0.0;
    for (I k = 0; k < nk; ++k) {
      double const av = *pa;
      double const den = *pb - av * scp;
      scp = *pc / den;
      sdp = (*pd - av * sdp) / den;
      *pcp = scp;
      *pdp = sdp;
      pa += a.sk, pb += b.sk, pc += c.sk, pd += d.sk, pcp += cp.sk, pdp += dp.sk;
    }
  }
  GK_INLINE void backward(I i, I j) const {
    double const *pcp = &cp(i, j, nk - 1), *pdp = &dp(i, j, nk - 1);
    double *px = &x(i, j, nk - 1);
    double xv = 0.0;
    for (I k = nk - 1; k >= 0; --k) {
      xv = *pdp - *pcp * xv;
      *px = xv;
      pcp -= cp.sk, pdp -= dp.sk, px -= x.sk;
    }
  }
};

enum class Part { both, fwd, bwd };

template <class T, Part P>
struct ColumnB {
  T t;
  using I = typename T::I;
  GK_INLINE void operator()(I i, I j) const {
    if constexpr (P != Part::bwd) t.forward(i, j);
    if constexpr (P != Part::fwd) t.backward(i, j);
  }
};

// Column-chunk executor: RangePolicy over blocks of B consecutive i-columns.
template <int B>
struct Chunk {
  template <class T, Part P>
  struct F {
    T t;
    int i0, i1, j0, nblk;
    KOKKOS_INLINE_FUNCTION void operator()(int w) const {
      T const tl = t;
      int const j = j0 + w / nblk, ib = i0 + (w % nblk) * B;
      int const n = ib + B <= i1 ? B : i1 - ib;
      if constexpr (P != Part::bwd) {
        if (n == B)
          tl.template forward_chunk<B>(ib, j, B);  // constant trip count for the full blocks
        else
          tl.template forward_chunk<B>(ib, j, n);
      }
      if constexpr (P != Part::fwd) {
        if (n == B)
          tl.template backward_chunk<B>(ib, j, B);
        else
          tl.template backward_chunk<B>(ib, j, n);
      }
    }
  };
  template <Part P, class T>
  static void run_part(Box2 d, T const &t) {
    int const nblk = (d.i1 - d.i0 + B - 1) / B;
    Kokkos::parallel_for(
        "gk_col_chunk",
        Kokkos::RangePolicy<Exec, Kokkos::IndexType<int>>(0, nblk * (d.j1 - d.j0)),
        F<T, P>{t, d.i0, d.i1, d.j0, nblk});
  }
};

// Adapts the per-column launch shapes to the same run_part interface.
template <class Shape>
struct PerColumn {
  template <Part P, class T>
  static void run_part(Box2 d, T const &t) {
    Shape::run(d, ColumnB<T, P>{t});
  }
};

template <class S, class A, class Launch, bool Split>
void solve(GKArgs const &g, A const &cp, A const &dp, A const &x) {
  using T = Tri<S, A>;
  T const t{field<A>(g, 0), field<A>(g, 1), field<A>(g, 2), field<A>(g, 3), cp, dp, x,
            static_cast<typename A::index_type>(g.nk)};
  Box2 const d{g.h, g.h + g.ni, g.h, g.h + g.nj};
  if constexpr (Split) {
    Launch::template run_part<Part::fwd>(d, t);
    Launch::template run_part<Part::bwd>(d, t);
  } else {
    Launch::template run_part<Part::both>(d, t);
  }
}

// caller-provided cp, dp (fields 4, 5), output x (field 6)
template <class S, class A, class Launch, bool Split = false>
void run_tri(GKArgs const &g) {
  solve<S, A, Launch, Split>(g, field<A>(g, 4), field<A>(g, 5), field<A>(g, 6));
}
// cp, dp temporaries, output x (field 4)
template <class S, class A, class Launch, class Tmp, bool Split = false>
void run_tritmp(GKArgs const &g) {
  std::size_t const n = gk::cart::tmp_size(g);
  Tmp t0(n), t1(n);
  solve<S, A, Launch, Split>(g, A::wrap(t0.data(), g, 0), A::wrap(t1.data(), g, 0),
                             field<A>(g, 4));
}

// Printed once per domain size: where compute rows start relative to a 256-byte boundary.
inline void report_alignment(GKArgs const &g) {
  static std::set<int> seen;
  if (!seen.insert(g.ni).second) return;
  auto const mod = [](double const *p) {
    return static_cast<int>(reinterpret_cast<std::uintptr_t>(p) % 256);
  };
  for (int f = 0; f < 5; ++f) {
    Ptr3<int> const a = field<Ptr3<int>>(g, f);
    std::fprintf(stderr,
                 "gk align ni=%d caller field %d: strides 1 %d %d, data %d, first compute point %d "
                 "(bytes mod 256)\n",
                 g.ni, f, a.sj, a.sk, mod(a.p), mod(&a(g.h, g.h, 0)));
  }
  // the buffers of the field-shaped pool variants
  TmpPool t0(gk::cart::tmp_size(g)), t1(gk::cart::tmp_size(g));
  for (double *buf : {t0.data(), t1.data()}) {
    Ptr3<int> const a = Ptr3<int>::wrap(buf, g, 0);
    std::fprintf(stderr, "gk align ni=%d pool buffer: data %d, first compute point %d\n", g.ni,
                 mod(buf), mod(&a(g.h, g.h, 0)));
  }
}

// cp, dp temporaries with a chosen layout. Compact: ni x nj x nk without padding, as gtfn allocates
// them; otherwise the strides of the caller fields (halo included, padded rows). The first compute
// point lies Off elements after a 256-byte boundary.
template <class S, class Launch, bool Compact, int Off, bool Split>
void run_tritmp_layout(GKArgs const &g) {
  using A = Ptr3<int>;
  report_alignment(g);
  int const sj = Compact ? g.ni : static_cast<int>(g.fs[0][1]);
  int const sk = Compact ? g.ni * g.nj : static_cast<int>(g.fs[0][2]);
  // 32 elements to reach the boundary and 32 for the offset
  std::size_t const n = static_cast<std::size_t>(sk) * g.nk + 64;
  TmpPool t0(n), t1(n);
  auto const place = [&](double *buf) {
    std::uintptr_t const up = (reinterpret_cast<std::uintptr_t>(buf) + 255) & ~std::uintptr_t{255};
    // origin shifted so that index (h, h, 0) is Off elements after the boundary
    return A{reinterpret_cast<double *>(up) + Off - (g.h + g.h * sj), sj, sk};
  };
  solve<S, A, Launch, Split>(g, place(t0.data()), place(t1.data()), field<A>(g, 4));
}

template <class S, class Launch, bool Split>
void add_layouts(std::string const &v) {
  std::string const base = Split ? v + ".split" : v;
  reg("tritmp", base + ".lay-c0", &run_tritmp_layout<S, Launch, true, 0, Split>);
  reg("tritmp", base + ".lay-c3", &run_tritmp_layout<S, Launch, true, 3, Split>);
  reg("tritmp", base + ".lay-p0", &run_tritmp_layout<S, Launch, false, 0, Split>);
  reg("tritmp", base + ".lay-p3", &run_tritmp_layout<S, Launch, false, 3, Split>);
}

template <class S, class A, class Launch>
void add(std::string const &v) {
  reg("tri", v, &run_tri<S, A, Launch>);
  reg("tritmp", v, &run_tritmp<S, A, Launch, TmpPool>);
}
template <class S, class A, class Launch>
void add_split(std::string const &v) {
  reg("tri", v + ".split", &run_tri<S, A, Launch, true>);
}
template <class S, class A, class Launch>
void add_tmp(std::string const &v) {
  reg("tritmp", v + ".alloc", &run_tritmp<S, A, Launch, TmpAlloc>);
  reg("tritmp", v + ".view", &run_tritmp<S, A, Launch, TmpView>);
  reg("tritmp", v + ".viewnoinit", &run_tritmp<S, A, Launch, TmpViewNoInit>);
}

template <class S, class A, class Launch>
void add_split_tmp(std::string const &v) {
  reg("tritmp", v + ".split", &run_tritmp<S, A, Launch, TmpPool, true>);
}

}  // namespace gk::scan

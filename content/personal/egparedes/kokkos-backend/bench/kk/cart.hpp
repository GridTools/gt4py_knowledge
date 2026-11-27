// Cartesian kernels, mirroring the gtfn_ir structure of the gtfn programs (same executions,
// same temporaries and domains, same recomputation after GTIR fusion).
#pragma once

#include "gk.hpp"

namespace gk::cart {

// ---------------------------------------------------------------- thin value iterator
template <class A>
struct It3 {
  using I = typename A::index_type;
  A a;
  I i, j, k;
  GK_INLINE double deref() const { return a(i, j, k); }
};
template <int Di, int Dj, int Dk, class A>
GK_INLINE It3<A> shift(It3<A> it) {
  it.i += Di;
  it.j += Dj;
  it.k += Dk;
  return it;
}

struct Iter {
  template <class It>
  GK_INLINE static double lap_it(It const &a) {
    return shift<1, 0, 0>(a).deref() + shift<-1, 0, 0>(a).deref() + shift<0, 1, 0>(a).deref() +
           shift<0, -1, 0>(a).deref() - a.deref() * 4.0;
  }
  template <class A, class I>
  GK_INLINE static double lap(A const &a, I i, I j, I k) {
    return lap_it(It3<A>{a, i, j, k});
  }
  template <class A, class I>
  GK_INLINE static double lap2(A const &f, I i, I j, I k) {
    It3<A> const a{f, i, j, k};
    auto const e = shift<1, 0, 0>(a), w = shift<-1, 0, 0>(a), n = shift<0, 1, 0>(a),
               s = shift<0, -1, 0>(a);
    return lap_it(e) + lap_it(w) + lap_it(n) + lap_it(s) -
           (e.deref() + w.deref() + n.deref() + s.deref() - a.deref() * 4.0) * 4.0;
  }
  template <class A, class I>
  GK_INLINE static double flx(A const &lpf, A const &af, I i, I j, I k) {
    It3<A> const lp{lpf, i, j, k}, a{af, i, j, k};
    double const f = shift<1, 0, 0>(lp).deref() - lp.deref();
    return (shift<1, 0, 0>(a).deref() - a.deref()) * f > 0.0 ? 0.0 : f;
  }
  template <class A, class I>
  GK_INLINE static double fly(A const &lpf, A const &af, I i, I j, I k) {
    It3<A> const lp{lpf, i, j, k}, a{af, i, j, k};
    double const f = shift<0, 1, 0>(lp).deref() - lp.deref();
    return (shift<0, 1, 0>(a).deref() - a.deref()) * f > 0.0 ? 0.0 : f;
  }
  template <class A, class I>
  GK_INLINE static double hout(A const &af, A const &cf, A const &fxf, A const &fyf, I i, I j,
                               I k) {
    It3<A> const a{af, i, j, k}, c{cf, i, j, k}, fx{fxf, i, j, k}, fy{fyf, i, j, k};
    return a.deref() - (fx.deref() - shift<-1, 0, 0>(fx).deref() + fy.deref() -
                        shift<0, -1, 0>(fy).deref()) *
                           c.deref();
  }
  template <class A, class I>
  GK_INLINE static double s1(A const &af, A const &bf, I i, I j, I k) {
    It3<A> const a{af, i, j, k}, b{bf, i, j, k};
    double const v = a.deref();
    return v * b.deref() + v;
  }
  template <class A, class I>
  GK_INLINE static double s2(A const &tf, I i, I j, I k) {
    It3<A> const t{tf, i, j, k};
    return (shift<1, 0, 0>(t).deref() + shift<-1, 0, 0>(t).deref()) * 0.5 +
           (shift<0, 1, 0>(t).deref() + shift<0, -1, 0>(t).deref()) * 0.25;
  }
};

struct Expl {
  template <class A, class I>
  GK_INLINE static double lap(A const &a, I i, I j, I k) {
    return a(i + 1, j, k) + a(i - 1, j, k) + a(i, j + 1, k) + a(i, j - 1, k) - a(i, j, k) * 4.0;
  }
  template <class A, class I>
  GK_INLINE static double lap2(A const &a, I i, I j, I k) {
    return lap(a, I(i + 1), j, k) + lap(a, I(i - 1), j, k) + lap(a, i, I(j + 1), k) +
           lap(a, i, I(j - 1), k) -
           (a(i + 1, j, k) + a(i - 1, j, k) + a(i, j + 1, k) + a(i, j - 1, k) - a(i, j, k) * 4.0) *
               4.0;
  }
  template <class A, class I>
  GK_INLINE static double flx(A const &lp, A const &a, I i, I j, I k) {
    double const f = lp(i + 1, j, k) - lp(i, j, k);
    return (a(i + 1, j, k) - a(i, j, k)) * f > 0.0 ? 0.0 : f;
  }
  template <class A, class I>
  GK_INLINE static double fly(A const &lp, A const &a, I i, I j, I k) {
    double const f = lp(i, j + 1, k) - lp(i, j, k);
    return (a(i, j + 1, k) - a(i, j, k)) * f > 0.0 ? 0.0 : f;
  }
  template <class A, class I>
  GK_INLINE static double hout(A const &a, A const &c, A const &fx, A const &fy, I i, I j, I k) {
    return a(i, j, k) -
           (fx(i, j, k) - fx(i - 1, j, k) + fy(i, j, k) - fy(i, j - 1, k)) * c(i, j, k);
  }
  template <class A, class I>
  GK_INLINE static double s1(A const &a, A const &b, I i, I j, I k) {
    double const v = a(i, j, k);
    return v * b(i, j, k) + v;
  }
  template <class A, class I>
  GK_INLINE static double s2(A const &t, I i, I j, I k) {
    return (t(i + 1, j, k) + t(i - 1, j, k)) * 0.5 + (t(i, j + 1, k) + t(i, j - 1, k)) * 0.25;
  }
};

// ---------------------------------------------------------------- one functor per execution
template <class S, class A>
struct LapB {
  A a, out;
  using I = typename A::index_type;
  GK_INLINE void operator()(I i, I j, I k) const { out(i, j, k) = S::lap(a, i, j, k); }
};
template <class S, class A>
struct Lap2B {
  A a, out;
  using I = typename A::index_type;
  GK_INLINE void operator()(I i, I j, I k) const { out(i, j, k) = S::lap2(a, i, j, k); }
};
template <class S, class A>
struct FlxB {
  A lp, a, out;
  using I = typename A::index_type;
  GK_INLINE void operator()(I i, I j, I k) const { out(i, j, k) = S::flx(lp, a, i, j, k); }
};
template <class S, class A>
struct FlyB {
  A lp, a, out;
  using I = typename A::index_type;
  GK_INLINE void operator()(I i, I j, I k) const { out(i, j, k) = S::fly(lp, a, i, j, k); }
};
template <class S, class A>
struct HoutB {
  A a, c, fx, fy, out;
  using I = typename A::index_type;
  GK_INLINE void operator()(I i, I j, I k) const { out(i, j, k) = S::hout(a, c, fx, fy, i, j, k); }
};
template <class S, class A>
struct S1B {
  A a, b, out;
  using I = typename A::index_type;
  GK_INLINE void operator()(I i, I j, I k) const { out(i, j, k) = S::s1(a, b, i, j, k); }
};
template <class S, class A>
struct S2B {
  A t, out;
  using I = typename A::index_type;
  GK_INLINE void operator()(I i, I j, I k) const { out(i, j, k) = S::s2(t, i, j, k); }
};

// interior compute domain, optionally extended by (lo, hi) points in i and j
inline Box3 box(GKArgs const &g, int ilo = 0, int ihi = 0, int jlo = 0, int jhi = 0) {
  return {g.h - ilo, g.h + g.ni + ihi, g.h - jlo, g.h + g.nj + jhi, 0, g.nk};
}
// temporaries have the full padded shape of field 0, so every accessor works unchanged
inline std::size_t tmp_size(GKArgs const &g) {
  return static_cast<std::size_t>(g.fs[0][2]) * g.fe[0][2];
}

template <class S, class A, class Shape>
void run_lap(GKArgs const &g) {
  Shape::run(box(g), LapB<S, A>{field<A>(g, 0), field<A>(g, 1)});
}
template <class S, class A, class Shape>
void run_lap2(GKArgs const &g) {
  Shape::run(box(g), Lap2B<S, A>{field<A>(g, 0), field<A>(g, 1)});
}
template <class S, class A, class Shape, class Tmp>
void run_hdiff(GKArgs const &g) {
  Tmp t0(tmp_size(g)), t1(tmp_size(g)), t2(tmp_size(g));
  A const a = field<A>(g, 0), c = field<A>(g, 1), out = field<A>(g, 2);
  A const lp = A::wrap(t0.data(), g, 0), fx = A::wrap(t1.data(), g, 0),
          fy = A::wrap(t2.data(), g, 0);
  Shape::run(box(g, 1, 1, 1, 1), LapB<S, A>{a, lp});
  Shape::run(box(g, 1, 0, 0, 0), FlxB<S, A>{lp, a, fx});
  Shape::run(box(g, 0, 0, 1, 0), FlyB<S, A>{lp, a, fy});
  Shape::run(box(g), HoutB<S, A>{a, c, fx, fy, out});
}
template <class S, class A, class Shape, class Tmp>
void run_tmpst(GKArgs const &g) {
  Tmp t0(tmp_size(g));
  A const a = field<A>(g, 0), b = field<A>(g, 1), out = field<A>(g, 2);
  A const t = A::wrap(t0.data(), g, 0);
  Shape::run(box(g, 1, 1, 1, 1), S1B<S, A>{a, b, t});
  Shape::run(box(g), S2B<S, A>{t, out});
}

// register one (style, accessor, shape) combination for all cartesian stencils (pooled temporaries)
template <class S, class A, class Shape>
void add(std::string const &v) {
  reg("lap", v, &run_lap<S, A, Shape>);
  reg("lap2", v, &run_lap2<S, A, Shape>);
  reg("hdiff", v, &run_hdiff<S, A, Shape, TmpPool>);
  reg("tmpst", v, &run_tmpst<S, A, Shape, TmpPool>);
}
// temporary strategies for one combination
template <class S, class A, class Shape>
void add_tmp(std::string const &v) {
  reg("hdiff", v + ".alloc", &run_hdiff<S, A, Shape, TmpAlloc>);
  reg("tmpst", v + ".alloc", &run_tmpst<S, A, Shape, TmpAlloc>);
  reg("hdiff", v + ".view", &run_hdiff<S, A, Shape, TmpView>);
  reg("tmpst", v + ".view", &run_tmpst<S, A, Shape, TmpView>);
  reg("hdiff", v + ".viewnoinit", &run_hdiff<S, A, Shape, TmpViewNoInit>);
  reg("tmpst", v + ".viewnoinit", &run_tmpst<S, A, Shape, TmpViewNoInit>);
}

}  // namespace gk::cart

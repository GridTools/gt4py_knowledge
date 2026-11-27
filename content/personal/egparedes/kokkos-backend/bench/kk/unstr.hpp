// Unstructured kernels: V2E neighbor sum with skip values, and the fused E2V -> V2E chain
// (one execution, as GTIR hands it to gtfn). Reductions are unrolled with a branch per skip.
#pragma once

#include "gk.hpp"

namespace gk::unstr {

inline constexpr int n_v2e = 6;

// ---------------------------------------------------------------- thin value iterator
template <class A>
struct It2 {
  A a;
  int h;  // -1: skip value, propagates through further shifts
  typename A::index_type k;
};
template <class A>
GK_INLINE bool can_deref(It2<A> const &it) {
  return it.h != -1;
}
template <class A>
GK_INLINE double deref(It2<A> const &it) {
  return it.a(it.h, it.k);
}
template <int N, class A, class T>
GK_INLINE It2<A> shift(It2<A> it, T const &tab) {
  if (it.h != -1) it.h = tab(it.h, N);
  return it;
}

struct Iter {
  template <class A, class T, class I, int... N>
  GK_INLINE static double v2e(A const &f, T const &tab, I h, I k,
                              std::integer_sequence<int, N...>) {
    It2<A> const e{f, static_cast<int>(h), k};
    double acc = 0.0;
    ((can_deref(shift<N>(e, tab)) ? acc = acc + deref(shift<N>(e, tab)) : acc), ...);
    return acc;
  }
  template <int N, class A, class T>
  GK_INLINE static double chain_step(double acc, It2<A> const &a, T const &v2e, T const &e2v) {
    It2<A> const e = shift<N>(a, v2e);
    // E2V has no skip values; the V2E skip is checked once, as in the generated gtfn code
    if (!can_deref(e)) return acc;
    return acc + (0.0 + deref(shift<0>(e, e2v)) + deref(shift<1>(e, e2v)));
  }
  template <class A, class T, class I, int... N>
  GK_INLINE static double chain(A const &f, T const &v2e, T const &e2v, I h, I k,
                                std::integer_sequence<int, N...>) {
    It2<A> const a{f, static_cast<int>(h), k};
    double acc = 0.0;
    ((acc = chain_step<N>(acc, a, v2e, e2v)), ...);
    return acc;
  }
};

struct Expl {
  template <class A, class T, class I, int... N>
  GK_INLINE static double v2e(A const &f, T const &tab, I h, I k,
                              std::integer_sequence<int, N...>) {
    double acc = 0.0;
    (
        [&] {
          int const e = tab(h, N);
          if (e != -1) acc += f(e, k);
        }(),
        ...);
    return acc;
  }
  template <class A, class T, class I, int... N>
  GK_INLINE static double chain(A const &f, T const &v2e, T const &e2v, I h, I k,
                                std::integer_sequence<int, N...>) {
    double acc = 0.0;
    (
        [&] {
          int const e = v2e(h, N);
          if (e != -1) acc += 0.0 + f(e2v(e, 0), k) + f(e2v(e, 1), k);
        }(),
        ...);
    return acc;
  }
};

// Follow-up variants that separate "iterator or explicit" from "how often the table is read".
// Iter (v2e) repeats shift() in the guard and in the access, exactly like the gtfn_ir code.
// Iter1 binds the shifted iterator once per neighbor (the shape the proposals sketch).
struct Iter1 {
  template <class A, class T, class I, int... N>
  GK_INLINE static double v2e(A const &f, T const &tab, I h, I k,
                              std::integer_sequence<int, N...>) {
    It2<A> const e{f, static_cast<int>(h), k};
    double acc = 0.0;
    (
        [&] {
          It2<A> const nb = shift<N>(e, tab);
          if (can_deref(nb)) acc += deref(nb);
        }(),
        ...);
    return acc;
  }
};
// IterLit (chain) re-shifts from the start for every access, exactly like the gtfn_ir code.
struct IterLit {
  template <int N, class A, class T>
  GK_INLINE static double step(double acc, It2<A> const &a, T const &v2e, T const &e2v) {
    return can_deref(shift<N>(a, v2e)) ? acc + (0.0 + deref(shift<0>(shift<N>(a, v2e), e2v)) +
                                                deref(shift<1>(shift<N>(a, v2e), e2v)))
                                       : acc;
  }
  template <class A, class T, class I, int... N>
  GK_INLINE static double chain(A const &f, T const &v2e, T const &e2v, I h, I k,
                                std::integer_sequence<int, N...>) {
    It2<A> const a{f, static_cast<int>(h), k};
    double acc = 0.0;
    ((acc = step<N>(acc, a, v2e, e2v)), ...);
    return acc;
  }
};
// Expl2 is explicit indexing that reads the table at every use, as the gtfn_ir code does.
struct Expl2 {
  template <class A, class T, class I, int... N>
  GK_INLINE static double v2e(A const &f, T const &tab, I h, I k,
                              std::integer_sequence<int, N...>) {
    double acc = 0.0;
    ((acc = tab(h, N) != -1 ? acc + f(tab(h, N), k) : acc), ...);
    return acc;
  }
  template <class A, class T, class I, int... N>
  GK_INLINE static double chain(A const &f, T const &v2e, T const &e2v, I h, I k,
                                std::integer_sequence<int, N...>) {
    double acc = 0.0;
    ((acc = v2e(h, N) != -1
                ? acc + (0.0 + f(e2v(v2e(h, N), 0), k) + f(e2v(v2e(h, N), 1), k))
                : acc),
     ...);
    return acc;
  }
};

template <class S, class A, class T>
struct V2EB {
  A in, out;
  T v2e;
  using I = typename A::index_type;
  GK_INLINE void operator()(I h, I k) const {
    out(h, k) = S::v2e(in, v2e, h, k, std::make_integer_sequence<int, n_v2e>{});
  }
};
template <class S, class A, class T>
struct ChainB {
  A in, out;
  T v2e, e2v;
  using I = typename A::index_type;
  GK_INLINE void operator()(I h, I k) const {
    out(h, k) = S::chain(in, v2e, e2v, h, k, std::make_integer_sequence<int, n_v2e>{});
  }
};

template <class S, class A, class T, class Shape>
void run_v2e(GKArgs const &g) {
  Shape::run(g.ni, g.nk, V2EB<S, A, T>{field<A>(g, 0), field<A>(g, 1), T::make(g, 0)});
}
template <class S, class A, class T, class Shape>
void run_chain(GKArgs const &g) {
  Shape::run(g.ni, g.nk,
             ChainB<S, A, T>{field<A>(g, 0), field<A>(g, 1), T::make(g, 0), T::make(g, 1)});
}

template <class S, class A, class T, class Shape>
void add(std::string const &v) {
  reg("v2e", v, &run_v2e<S, A, T, Shape>);
  reg("chain", v, &run_chain<S, A, T, Shape>);
}

}  // namespace gk::unstr

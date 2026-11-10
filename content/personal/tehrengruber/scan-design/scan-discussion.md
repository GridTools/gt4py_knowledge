## 2026.10.30 H, E, S

TODOs:

- Let's find an example of an inclusive scan where the init value is a good concept (on any level).
```
@gtx.scan_operator(axis=dims.KDim, forward=True, init=(wpfloat("0.0"), wpfloat("0.0")))
def _solve_tridiagonal_matrix_forward_sweep(
    state_kminus1: tuple[wpfloat, wpfloat],
    a: wpfloat,
    b: wpfloat,
    c: wpfloat,
    d: wpfloat,
) -> tuple[wpfloat, wpfloat]:
    q_kminus1, d_prime_kminus1 = state_kminus1
    normalization = wpfloat("1.0") / (b + a * q_kminus1)
    return (wpfloat("0.0") - c) * normalization, (d - a * d_prime_kminus1) * normalization
```
Look if there is (by research elsewhere) if there is a deductive answer to the question if the init value in the inclusive scan can be removed without losing expressiveness.
- Let's take a look at how the pattern we target here are expressed in Jax, Torch, maybe even Fortran and how do they compare.


### Scratchpad
Exclusive scan using in the init what was previously an input value
```
@scan_pass
def scan_pass(out: KView[float], inp: KView[float]):
  return inp
  
@fo
def fo():
  scan_pass(inp, init=2*inp[K(0)])
```
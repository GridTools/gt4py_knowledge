"""Typing probe for the `V2E.Local` question (B1 in the adversarial review).

Run with `mypy --strict --python-version 3.12` and `pyright --pythonversion 3.12`.
Expected: P1 accepts the explicit nested `Local` and rejects the cross-connectivity
mismatch on line `p1(Field[E, E_V2V.Local]())`; P2 rejects the generated `ClassVar` form
as "not valid as a type"; P3 rejects each spelling where the other is expected.
pyright additionally reports "Class definition for V_E2E depends on itself" on the
string forward reference in the base subscription, which is why the proposal does
not pass `Local` through the bases.
"""

from __future__ import annotations

import sys
import typing
from collections.abc import Iterator, Mapping
from types import MappingProxyType, get_original_bases


class DimensionIndex(int): ...

class LocalDimensionIndex(int): ...

type MultiLevelDimensionIndex[D: DimensionIndex, L: LocalDimensionIndex] = tuple[D, *tuple[L, ...], L]   #  (V(0), LocalV_E(2))


class StaticMultiLevelMapping[D: DimensionIndex, L: LocalDimensionIndex, V](
    Mapping[MultiLevelDimensionIndex[D, L], V]
):
    """Read-only mapping fixed at class-definition time; one instance per class.

    `data` uses plain ints; keys and values are converted to the D, L, V types
    given in the subclass's base subscription (every tail element becomes L).
    """

    __slots__ = ()
    _data: Mapping[MultiLevelDimensionIndex[D, L], V]
    _instance: typing.ClassVar[typing.Any]

    def __init_subclass__(
        cls, *, data: Mapping[tuple[int, ...], int] | None = None, **kwargs: typing.Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        if data is not None:
            dim, local, value = cls._resolve_type_args()
            cls._data = MappingProxyType(
                {(dim(k[0]), *map(local, k[1:])): value(v) for k, v in data.items()}
            )

    @classmethod
    def _resolve_type_args(cls) -> tuple[type, type, type]:
        base = next(
            b for b in get_original_bases(cls)
            if typing.get_origin(b) is StaticMultiLevelMapping
        )
        # cls isn't bound in its module yet, so forward refs like "V_V2V.Local"
        # need it injected explicitly.
        ns = vars(sys.modules[cls.__module__]) | {cls.__name__: cls}
        args = tuple(eval(a, ns) if isinstance(a, str) else a for a in typing.get_args(base))
        assert len(args) == 3, args
        return args

    def __new__(cls) -> typing.Self:
        if "_instance" not in cls.__dict__:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __getitem__(self, key: MultiLevelDimensionIndex[D, L]) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[MultiLevelDimensionIndex[D, L]]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class V(DimensionIndex): ...

class E(DimensionIndex): ...


class V_E2E(StaticMultiLevelMapping[V, "V_E2E.Local", E], data={(0, 0): 1, (0, 1): 2}):
    class Local(LocalDimensionIndex): ...


class E_V2V(StaticMultiLevelMapping[E, "E_V2V.Local", V], data={(0, 0): 1, (0, 1): 2}):
    class Local(LocalDimensionIndex): ...



# ---------------- probes appended by reviewer ----------------
class Field[*Ds]: ...

# P1: explicitly declared nested Local used in an annotation
def p1(a: Field[V, V_E2E.Local]) -> None: ...
p1(Field[V, V_E2E.Local]())
p1(Field[E, E_V2V.Local]())   # expected error: distinct locals

# P2: generated Local (ClassVar on the base), as the proposal's default form
class Base2:
    Local: typing.ClassVar[type[LocalDimensionIndex]]
    def __init_subclass__(cls, **kw: typing.Any) -> None:
        cls.Local = type("Local", (LocalDimensionIndex,), {})
class G2E(Base2): ...
def p2(a: Field[V, G2E.Local]) -> None: ...   # does mypy accept this annotation?

# P3: reconciliation Local[C] == C.Local via __class_getitem__
class Local[C]:
    def __class_getitem__(cls, item: typing.Any) -> typing.Any:
        return item.Local
def p3a(a: Field[V, V_E2E.Local]) -> None: ...
def p3b(a: Field[V, Local[V_E2E]]) -> None: ...
p3a(Field[V, Local[V_E2E]]())    # same type for the checker?
p3b(Field[V, V_E2E.Local]())

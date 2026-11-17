"""Enumerate functions without generating duplicates.

This is an alternative to ``duplicate_checker.main``. That routine builds every
tree of a given complexity and then removes duplicates with sympy. Here each
function (or family of functions, if it has free parameters) is built once, at
the lowest complexity at which it can be written, and functions are identified
numerically, so nothing needs to be simplified symbolically.

Method
------

1. *Bottom-up over complexity.* If a function's shortest tree has ``c`` nodes,
   every subtree of that tree is itself a shortest tree (otherwise the whole
   tree could be shortened). So all functions of complexity ``c`` are found by
   applying the operators to the functions already kept at lower
   complexities, and never to their duplicates. Commutative operators are
   applied to each unordered pair only once.

2. *Parameters as typed constants.* A subtree containing parameters but no
   variables is a constant whose possible values form an interval. As every
   parameter appears once, interval arithmetic gives that interval exactly:
   ``a``, ``a+a``, ``inv(a)`` and ``log_abs(a)`` are all "any real", while
   ``exp(a)`` and ``sqrt_abs(a)`` are "any non-negative real". Within a
   function each such subtree is one *slot* taking values in its interval.

3. *Numerical keys.* Functions are evaluated at fixed sample points of the
   variables. Each function gets one or more keys, which are invariants of
   the function (or family): two functions are the same if any of their keys
   agree.

   * Without parameters, the key is the values themselves.
   * A family that fills an affine subspace of function space after applying
     ``f``, ``1/f`` or ``log|f|`` to its values is keyed by that subspace
     (with the sign of its members for ``log|f|``). This recognises any
     reparametrisation, e.g. ``a0*x + a1`` and ``a0*(x + a1)`` (both every
     ``u*x + v``), or ``a0*pow(a1, x)`` and ``pow(a0, x)/a1`` (``log|f|`` is
     every ``u + v*x``). Whether the family fills the subspace, rather than
     part of it (``x + exp(a0)``), is tested by solving for target functions
     at increasing distances along random directions in the subspace.
   * A family with as many independent directions as parameters is also keyed
     by its set of values over a fixed set of parameter values, closed under
     ``b -> -b`` and ``b -> 1/b`` of each parameter and under permutations of
     the parameters, which recognises e.g. ``pow(x + a0, a1)`` and
     ``pow(a0 - x, a1)``.
   * A family with fewer independent directions than parameters (its
     Jacobian is rank deficient, e.g. ``pow(a0, a1*x)``) and no subspace key
     is a reparametrisation of a smaller family, and is dropped.

Values are compared on a logarithmic grid with relative spacing ``1e-9``
(``1e-6`` for subspaces), on two staggered grids so that rounding never
separates two equal functions.

The catalogue written for each complexity uses the same files as
``duplicate_checker`` (``all_equations``, ``unique_equations``, ``matches``,
``inv_subs`` and ``aifeyn``), so the fitting routines can be run on it
unchanged. As every function appears once, ``all_equations`` and
``unique_equations`` are the same list, ``matches`` is the identity and there
are no inverse substitutions. Each function is listed only at its lowest
complexity, whereas ``duplicate_checker`` lists it at every complexity at
which it can be written.

Limitations: equality is tested numerically at a finite set of points, so it
is not a proof. Families related by a reparametrisation outside those above
(e.g. ``pow(x, a0) - x*a1`` and ``x*(pow(x, a0) - a1)``, related by
``a0 -> a0 + 1``) are kept separately, and a family filling all but a thin
region of an affine subspace (e.g. ``(a0 - x)*(a1 - x)``) may be merged into
the full family containing it.
"""

import hashlib
import itertools
import os
import sys
import zlib

import numpy as np
from mpi4py import MPI

from esr.fitting.sympy_symbols import function_set_dirname, register_variables
from esr.generation import generator, utils
from esr.generation.duplicate_checker import _apply_variables, builtin_basis_functions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N_POINTS = 24  # sample points per variable
_REL_STEP = 1e-9  # relative spacing of the value grid
_REL_STEP_SUBSPACE = 1e-6  # relative spacing of the grid for affine subspaces
_ZERO = 1e-11  # values smaller than this are taken to be zero
_RANK_TOL = 1e-6  # relative singular value cut for ranks
_MAX_GROUP = 150000  # most reparametrisations tried for one family

INF = np.inf
REAL = (-INF, INF)

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

UNARY = {
    "inv": lambda a: 1.0 / a,
    "square": lambda a: a * a,
    "cube": lambda a: a * a * a,
    "sqrt_abs": lambda a: np.sqrt(np.abs(a)),
    "log_abs": lambda a: np.log(np.abs(a)),
    "log10_abs": lambda a: np.log10(np.abs(a)),
    "exp": np.exp,
    "tenexp": lambda a: np.power(10.0, a),
    "sin": np.sin,
    "cos": np.cos,
}

BINARY = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "pow": lambda a, b: np.power(np.abs(a), b),
}

COMMUTATIVE = {"+", "*"}


# ---------------------------------------------------------------------------
# Interval arithmetic for constant subtrees
# ---------------------------------------------------------------------------


def _iv_abs(a):
    lo, hi = a
    if lo >= 0:
        return a
    if hi <= 0:
        return (-hi, -lo)
    return (0.0, max(-lo, hi))


def _iv_monotone(f, a):
    with np.errstate(all="ignore"):
        return (float(f(a[0])), float(f(a[1])))


def _iv_mul(a, b):
    with np.errstate(all="ignore"):
        p = np.array([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]])
    p[np.isnan(p)] = 0.0  # 0 * inf
    return (float(p.min()), float(p.max()))


def _iv_inv(a):
    lo, hi = a
    if lo == 0 and hi == 0:
        return None
    with np.errstate(all="ignore"):
        if lo > 0 or hi < 0:
            return (1.0 / hi, 1.0 / lo)
        if lo == 0:
            return (1.0 / hi, INF)
        if hi == 0:
            return (-INF, 1.0 / lo)
    return REAL


def _iv_trig(f, a, shift):
    lo, hi = a
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo >= 2 * np.pi:
        return (-1.0, 1.0)
    # Extremes are at the end points or where f' = 0, i.e. shift + k pi
    k0 = np.ceil((lo - shift) / np.pi)
    k1 = np.floor((hi - shift) / np.pi)
    pts = [lo, hi] + [shift + k * np.pi for k in np.arange(k0, k1 + 1)]
    v = f(np.array(pts))
    return (float(v.min()), float(v.max()))


def _iv_unary(op, a):
    if op == "inv":
        return _iv_inv(a)
    if op == "square":
        b = _iv_abs(a)
        return (b[0] * b[0], b[1] * b[1])
    if op == "cube":
        return _iv_monotone(lambda t: t**3, a)
    if op == "sqrt_abs":
        return _iv_monotone(np.sqrt, _iv_abs(a))
    if op == "log_abs":
        return _iv_monotone(np.log, _iv_abs(a))
    if op == "log10_abs":
        return _iv_monotone(np.log10, _iv_abs(a))
    if op == "exp":
        return _iv_monotone(np.exp, a)
    if op == "tenexp":
        return _iv_monotone(lambda t: np.power(10.0, t), a)
    if op == "sin":
        return _iv_trig(np.sin, a, np.pi / 2)
    if op == "cos":
        return _iv_trig(np.cos, a, 0.0)
    raise ValueError(f"Unknown unary operator {op!r}")


def _iv_binary(op, a, b):
    if op == "+":
        return (a[0] + b[0], a[1] + b[1])
    if op == "-":
        return (a[0] - b[1], a[1] - b[0])
    if op == "*":
        return _iv_mul(a, b)
    if op == "/":
        ib = _iv_inv(b)
        return None if ib is None else _iv_mul(a, ib)
    if op == "pow":
        # |a|^b = exp(b log|a|), and b, a are independent
        loga = _iv_monotone(np.log, _iv_abs(a))
        return _iv_monotone(np.exp, _iv_mul(b, loga))
    raise ValueError(f"Unknown binary operator {op!r}")


def _round_interval(iv):
    if iv is None or np.isnan(iv[0]) or np.isnan(iv[1]):
        return None
    return tuple(float(f"{v:.12g}") if np.isfinite(v) else float(v) for v in iv)


# ---------------------------------------------------------------------------
# Parameter values used for keys
# ---------------------------------------------------------------------------

# Each slot j of a family is given the value phi(b_j), where b_j is the j-th
# entry of a fixed list of generic real numbers and phi maps the real line onto
# the slot's interval: the identity for the real line, |b| for [0, inf), and
# so on (see ``_phi``). For a half-line phi only depends on |b|, so a family
# that uses a real slot only through its absolute value (e.g. ``pow(a0, x)`` =
# ``|a0|^x``) takes the same values as the same family written with a
# non-negative slot (e.g. ``pow(exp(a0), x)``).
_POOL_RNG = np.random.RandomState(20261116)
_POOL_REAL = _POOL_RNG.uniform(0.3, 3.0, 32) * np.where(np.arange(32) % 2, -1, 1)


def _phi(iv, b):
    """Map the real number(s) ``b`` into the interval ``iv``."""
    lo, hi = iv
    if lo == -INF and hi == INF:
        return b
    m = np.abs(b)
    if hi == INF:
        return lo + m
    if lo == -INF:
        return hi - m
    return lo + (hi - lo) * m / (1.0 + m)


def _generic_values(ivs):
    """The fixed generic parameter vector of a family with slot intervals ``ivs``."""
    return np.array([_phi(iv, _POOL_REAL[j]) for j, iv in enumerate(ivs)])


def _to_slot(iv, u):
    """Map an unconstrained number ``u`` into the interval ``iv``.

    A real slot is ``sinh(u)``, so that both very large and very small
    magnitudes are reached by moderate steps in ``u`` (``log|sinh(u)|`` is
    close to linear for large ``|u|``).
    """
    lo, hi = iv
    if lo == -INF and hi == INF:
        return np.sinh(u)
    if hi == INF:
        return lo + np.exp(u)
    if lo == -INF:
        return hi - np.exp(u)
    return lo + (hi - lo) / (1.0 + np.exp(-u))


def _from_slot(iv, t):
    lo, hi = iv
    if lo == -INF and hi == INF:
        return np.arcsinh(t)
    if hi == INF:
        return np.log(t - lo)
    if lo == -INF:
        return np.log(hi - t)
    s = (t - lo) / (hi - lo)
    return np.log(s / (1 - s))


def _random_values(ivs, rng, m):
    """``m`` random parameter vectors, shape ``(m, len(ivs))``."""
    u = rng.normal(0.0, 1.2, (m, len(ivs)))
    return np.column_stack([_to_slot(iv, u[:, j]) for j, iv in enumerate(ivs)])


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _quantise(v, rel, zero):
    """Two staggered integer encodings of the values ``v`` (along the last axis).

    Each value is encoded by its sign (or nan/inf/zero) and ``log|v| / rel``
    rounded to an integer, the second encoding being offset by half a step.
    Two equal values that differ by rounding error get the same code in at
    least one of the two encodings.
    """
    v = np.asarray(v, dtype=float)
    with np.errstate(all="ignore"):
        a = np.abs(v)
        nz = np.isfinite(v) & (a >= zero)
        logv = np.where(nz, np.log(np.where(nz, a, 1.0)), 0.0) / rel
    code = np.zeros(v.shape, dtype=np.int64)
    code[nz & (v > 0)] = 1
    code[nz & (v < 0)] = 2
    code[np.isnan(v)] = 3
    code[np.isposinf(v)] = 4
    code[np.isneginf(v)] = 5
    q0 = np.round(logv).astype(np.int64)
    q1 = np.round(logv + 0.5).astype(np.int64)
    return (
        np.concatenate([code, q0], axis=-1),
        np.concatenate([code, q1], axis=-1),
    )


def _rows_to_bytes(arr):
    return [r.tobytes() for r in np.atleast_2d(arr)]


def _rank(mat, tol=_RANK_TOL):
    if mat.size == 0:
        return 0
    s = np.linalg.svd(mat, compute_uv=False)
    if s[0] == 0 or not np.isfinite(s[0]):
        return 0
    return int(np.sum(s > tol * s[0]))


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


class Func:
    """A function or family of functions, stored as its representative tree.

    Attributes:
        :c (int): complexity (number of nodes of the tree)
        :op (str): operator at the root, or the label of a leaf
        :kids (tuple): the child ``Func`` objects (empty for a leaf)
        :has_var (bool): whether the tree contains a variable
        :has_param (bool): whether the tree contains a parameter
        :kind (str): ``"fixed"`` (no parameters), ``"const"`` (parameters but
            no variables) or ``"family"`` (both)
        :V (numpy.ndarray): values at the sample points (``kind == "fixed"``)
        :iv (tuple): interval of values (``kind == "const"``)
        :ivs (list): the interval of each slot (``kind == "family"``)
        :ev (callable): maps slot values of shape ``(m, k)`` to function values
            broadcastable to ``(m, N_POINTS)`` (``kind == "family"``)
        :linear (bool): whether the function is affine in its parameters (set
            when its key is computed)
    """

    __slots__ = (
        "V",
        "_aifeyn",
        "c",
        "ev",
        "has_param",
        "has_var",
        "iv",
        "ivs",
        "kids",
        "kind",
        "linear",
        "op",
    )

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)

    def slot_form(self):
        """Number of slots, their intervals and the evaluator, as a child."""
        if self.kind == "fixed":
            V = self.V
            return 0, [], lambda th: V
        if self.kind == "const":
            return 1, [self.iv], lambda th: th[:, 0:1]
        return len(self.ivs), self.ivs, self.ev

    def prefix(self, counter=None):
        """Node labels in prefix order, with parameters named a0, a1, ..."""
        if counter is None:
            counter = [0]
        if not self.kids:
            if self.op == "a":
                counter[0] += 1
                return [f"a{counter[0] - 1}"]
            return [self.op]
        out = [self.op]
        for k in self.kids:
            out += k.prefix(counter)
        return out

    def string(self, counter=None):
        """The function as a string that the fitting routines can read."""
        if counter is None:
            counter = [0]
        if not self.kids:
            if self.op == "a":
                counter[0] += 1
                return f"a{counter[0] - 1}"
            return f"({self.op})" if self.op.startswith("-") else self.op
        if len(self.kids) == 1:
            return f"{self.op}({self.kids[0].string(counter)})"
        left = self.kids[0].string(counter)
        right = self.kids[1].string(counter)
        if self.op == "pow":
            return f"pow({left}, {right})"
        if len(self.kids[0].kids) == 2:
            left = f"({left})"
        if len(self.kids[1].kids) == 2:
            right = f"({right})"
        return f"{left}{self.op}{right}"

    def aifeyn(self):
        if self._aifeyn is None:
            labels = self.prefix()
            nparam = sum(1 for t in labels if t[0] == "a" and t[1:].isdigit())
            self._aifeyn = float(
                generator.aifeyn_complexity(labels, [f"a{i}" for i in range(nparam)])
            )
        return self._aifeyn


def _leaf(label, var_points):
    f = Func()
    f.c, f.op, f.kids = 1, label, ()
    f.has_var = label in var_points
    f.has_param = label == "a"
    if f.has_var:
        f.kind, f.V = "fixed", var_points[label]
    elif f.has_param:
        f.kind, f.iv = "const", REAL
    else:
        f.kind, f.V = "fixed", np.full(N_POINTS, float(label))
    return f


def _combine(op, kids):
    """Build the Func ``op(*kids)``, or None if it is undefined everywhere."""
    f = Func()
    f.op, f.kids = op, tuple(kids)
    f.c = 1 + sum(k.c for k in kids)
    f.has_var = any(k.has_var for k in kids)
    f.has_param = any(k.has_param for k in kids)
    fn = UNARY[op] if len(kids) == 1 else BINARY[op]

    if not f.has_param:
        f.kind = "fixed"
        with np.errstate(all="ignore"):
            f.V = np.broadcast_to(fn(*[k.V for k in kids]), (N_POINTS,)).copy()
        if not np.any(np.isfinite(f.V)):
            return None
        # A function of the variables that is constant (e.g. x/x) is a number
        if (
            f.has_var
            and np.all(np.isfinite(f.V))
            and np.ptp(f.V) <= 1e-12 * max(1.0, np.max(np.abs(f.V)))
        ):
            f.has_var = False
        return f

    if not f.has_var:
        ivs = [k.iv if k.kind == "const" else (k.V[0], k.V[0]) for k in kids]
        iv = _iv_unary(op, ivs[0]) if len(kids) == 1 else _iv_binary(op, *ivs)
        iv = _round_interval(iv)
        if iv is None:
            return None
        if iv[0] == iv[1]:
            if not np.isfinite(iv[0]):
                return None
            f.kind, f.V = "fixed", np.full(N_POINTS, iv[0])
            return f
        f.kind, f.iv = "const", iv
        return f

    f.kind = "family"
    forms = [k.slot_form() for k in kids]
    f.ivs = [iv for form in forms for iv in form[1]]
    if len(kids) == 1:
        g = forms[0][2]
        f.ev = lambda th: fn(g(th))
    else:
        k1, g1, g2 = forms[0][0], forms[0][2], forms[1][2]
        f.ev = lambda th: fn(g1(th[:, :k1]), g2(th[:, k1:]))
    return f


def _family_eval(f, th):
    with np.errstate(all="ignore"):
        if f.kind == "const":
            return np.broadcast_to(th[:, 0:1], (th.shape[0], N_POINTS))
        return np.broadcast_to(f.ev(th), (th.shape[0], N_POINTS))


def _family_ivs(f):
    return [f.iv] if f.kind == "const" else f.ivs


def _jacobian_rank(f, t):
    """Rank of the Jacobian of the family ``f`` at the parameters ``t``."""
    k = len(t)
    step = 1e-6 * np.maximum(1.0, np.abs(t))
    th = np.repeat(t[None], 2 * k, axis=0)
    for j in range(k):
        th[2 * j, j] += step[j]
        th[2 * j + 1, j] -= step[j]
    vals = _family_eval(f, th)
    J = (vals[0::2] - vals[1::2]) / step[:, None]  # (k, N)
    J = J[:, np.all(np.isfinite(J), axis=0)]
    norms = np.linalg.norm(J, axis=1)
    if not np.any(norms > 0):
        return 0
    return _rank(J[norms > 0] / norms[norms > 0, None])


_TRANSFORMS = {
    "id": lambda v: v,
    "inv": lambda v: 1.0 / v,
    "log": lambda v: np.log(np.abs(v)),
}


class _Keyer:
    """Computes the keys of functions; see the module docstring."""

    def __init__(self, seed=12345):
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.probe = rng.normal(size=(2, N_POINTS))
        self.n_deficient = 0

    def fixed_key(self, V):
        if not np.any(np.isfinite(V)):
            return None
        k0, k1 = _quantise(V, _REL_STEP, _ZERO)
        return (b"F" + k0.tobytes(), b"F" + k1.tobytes())

    def key(self, f):
        """The keys of ``f``, or None if ``f`` should be dropped. Also sets
        ``f.linear``.

        Each key is a pair (one per value grid) and is an invariant of the
        family: two functions with a key in common are the same. A family can
        have several keys (an affine-subspace key and a permutation/inversion
        key), and two functions are the same if any of their keys agree.
        """
        f.linear = f.kind == "fixed"
        with np.errstate(all="ignore"):
            keys = self._key(f)
        if keys is None or len(keys) == 0:
            return None
        return keys

    def _key(self, f):
        if f.kind == "fixed":
            key = self.fixed_key(f.V)
            return None if key is None else [key]
        ivs = _family_ivs(f)
        k = len(ivs)
        t0 = _generic_values(ivs)
        V0 = _family_eval(f, t0[None])[0]
        fin = np.isfinite(V0)
        if fin.sum() < k + 3:
            return None

        # Number of independent directions of the family: the rank of its
        # Jacobian at a generic point, i.e. the largest rank found at a few
        # points (at any one point, values that overflow or underflow can
        # hide a direction)
        rng = np.random.RandomState(self.seed)
        r = 0
        for t in np.vstack([t0[None], _random_values(ivs, rng, 3)]):
            r = max(r, _jacobian_rank(f, t))
            if r == k:
                break
        if r == 0:
            key = self.fixed_key(V0)
            return None if key is None else [key]

        keys = []
        S = _family_eval(f, _random_values(ivs, rng, k + 6))
        for name in _TRANSFORMS:
            key = self._affine_key(f, ivs, t0, V0, S, fin, r, name, rng)
            if key is not None:
                keys.append(key)
                break
        if r == k:
            keys.append(self._orbit_key(f, ivs))
        elif not keys:
            # More parameters than independent directions, and not an affine
            # family: a reparametrisation of a family with fewer parameters
            self.n_deficient += 1
            return None
        return keys

    def _affine_key(self, f, ivs, t0, V0, S, fin, r, name, rng):
        """Key of the family if, after applying the transform ``name`` to its
        values, it fills an affine subspace; otherwise None."""
        T = _TRANSFORMS[name]
        TV0 = T(V0)
        finT = fin & np.isfinite(TV0)
        if finT.sum() < r + 3:
            return None
        TS = T(S[:, finT])
        ok = np.all(np.isfinite(TS), axis=1)
        sign = b""
        if name == "log":
            # log|F| forgets the sign, so every member must have one sign
            # throughout, and if members of both signs occur the family must
            # be closed under F -> -F
            sg = np.sign(np.vstack([V0[finT], S[ok][:, finT]]))
            if np.any(sg != sg[:, :1]):
                return None
            signs = set(sg[:, 0])
            if len(signs) == 2:
                if not self._reaches(f, ivs, t0, fin, -S[ok][:3][:, fin], "id", rng):
                    return None
                sign = b"+-"
            else:
                sign = b"+" if signs == {1.0} else b"-"
        D = TS[ok] - TV0[finT]
        if D.shape[0] < r + 2:
            return None
        scale = np.max(np.abs(D))
        if not (scale > 0 and np.isfinite(scale)) or _rank(D / scale, tol=1e-8) != r:
            return None

        # Is the family affine in its parameters (then it certainly fills the
        # subspace if the parameters are unrestricted)?
        affine = False
        if name == "id" and all(iv == REAL for iv in ivs):
            ta, tb = _random_values(ivs, rng, 2)
            vals = _family_eval(f, np.array([ta, tb, ta + tb - t0]))
            affine = np.allclose(
                (vals[0] + vals[1] - V0)[fin], vals[2][fin], rtol=1e-9, atol=1e-9
            )
        if not affine and not self._fills(f, ivs, t0, TV0, finT, D, r, name, rng):
            return None
        if name == "id":
            f.linear = affine
        return self._subspace_key(TV0, finT, D, r, name.encode() + sign)

    def _fills(self, f, ivs, t0, TV0, fin, D, r, name, rng):
        """Whether the family fills the affine subspace spanned by ``D``.

        Along a few random directions from ``T(F(theta*))`` within the
        subspace, solve ``T(F(theta)) = y`` for targets ``y`` at increasing
        distance. Each solve starts from the solution for the previous target
        on the same direction, and from the members of a large random sample
        of the family (with heavy tails, so that it includes extreme
        parameters) that are closest to the target. A family that fills the
        subspace is followed to every target; a restricted one fails where a
        direction leaves its range.
        """
        T = _TRANSFORMS[name]
        k = len(ivs)
        _, _, vt = np.linalg.svd(D, full_matrices=False)
        Q = vt[:r]
        base = np.median(np.linalg.norm(D, axis=1))

        # Random sample of the family, in coordinates of the subspace
        u_pool = np.vstack(
            [rng.normal(0.0, 2.0, (1000, k)), rng.normal(0.0, 8.0, (1000, k))]
        )
        th = np.column_stack([_to_slot(iv, u_pool[:, j]) for j, iv in enumerate(ivs)])
        TF = T(_family_eval(f, th)[:, fin])
        good = np.all(np.isfinite(TF), axis=1)
        u_pool = u_pool[good]
        c_pool = (TF[good] - TV0[fin]) @ Q.T  # (n, r)

        n_dir, n_near, n_rand = 6, 3, 1
        # Each near sample is also tried with the sign of each real slot
        # flipped, which puts a start on the far side of a missing hyperplane
        # (e.g. a target of (a1 - x)/a0 near zero slope may need a0 large and
        # of the opposite sign to the nearest samples)
        flips = [j for j, iv in enumerate(ivs) if iv == REAL]
        n_start = 1 + n_near * (1 + len(flips)) + n_rand
        d = rng.normal(0.0, 1.0, (n_dir, r))
        d /= np.linalg.norm(d, axis=1)[:, None]
        best = np.tile([_from_slot(iv, t) for iv, t in zip(ivs, t0)], (n_dir, 1))
        for scale in [0.3, 1.0, 3.0, 10.0]:
            c_target = scale * base * d  # (n_dir, r)
            Y = TV0[fin] + c_target @ Q
            starts = [best[:, None, :]]
            if len(c_pool) >= n_near:
                dist = np.linalg.norm(c_pool[None] - c_target[:, None], axis=2)
                near = u_pool[np.argsort(dist, axis=1)[:, :n_near]]
            else:
                near = rng.normal(0.0, 1.5, (n_dir, n_near, k))
            starts.append(near)
            for j in flips:
                flipped = near.copy()
                flipped[:, :, j] *= -1
                starts.append(flipped)
            starts.append(rng.normal(0.0, 1.5, (n_dir, n_rand, k)))
            starts = np.concatenate(starts, axis=1).reshape(-1, k)
            Yr = np.repeat(Y, n_start, axis=0)
            # Targets that need a parameter near the end of its range (e.g.
            # a0 -> 0 with a1 -> inf) are only approached slowly, so a target
            # counts as hit when the residual is small relative to it
            tol = 1e-5 * (np.linalg.norm(Yr, axis=1) + 1)
            res, U = _batched_least_squares(
                f, ivs, fin, starts, Yr, T, tol=tol, group=n_start
            )
            res = res.reshape(n_dir, n_start)
            if not (res < tol.reshape(n_dir, -1)).any(axis=1).all():
                return False
            best = U.reshape(n_dir, n_start, k)[np.arange(n_dir), res.argmin(axis=1)]
        return True

    def _reaches(self, f, ivs, t0, fin, Y, name, rng):
        """Whether ``T(F(theta)) = y`` can be solved for every row ``y`` of
        ``Y`` (at the points ``fin``), trying several starting points."""
        u0 = np.array([_from_slot(iv, t) for iv, t in zip(ivs, t0)])
        starts = np.vstack(
            [
                u0,
                rng.normal(0.0, 1.5, (4, len(ivs))),
                rng.normal(0.0, 8.0, (3, len(ivs))),
            ]
        )
        n_start = len(starts)
        for rows in [slice(0, 3), slice(3, None)]:
            Yb = Y[rows]
            if len(Yb) == 0:
                continue
            U = np.tile(starts, (len(Yb), 1))
            Yb = np.repeat(Yb, n_start, axis=0)
            tol = 1e-5 * (np.linalg.norm(Yb, axis=1) + 1)
            res, _ = _batched_least_squares(
                f, ivs, fin, U, Yb, _TRANSFORMS[name], tol=tol, group=n_start
            )
            if not (res < tol).reshape(-1, n_start).any(axis=1).all():
                return False
        return True

    def _subspace_key(self, V0, fin, D, r, tag):
        _, _, vt = np.linalg.svd(D, full_matrices=False)
        Q = vt[:r]
        v = V0[fin]
        offset = v - (Q @ v) @ Q
        proj = (self.probe[:, fin] @ Q.T) @ Q
        zero = 1e-8 * max(1.0, np.max(np.abs(v)))
        parts = [offset] + list(proj)
        k0, k1 = zip(*[_quantise(p, _REL_STEP_SUBSPACE, zero) for p in parts])
        head = b"A" + tag + b":" + bytes([r]) + np.packbits(fin).tobytes()
        return (
            head + b"".join(a.tobytes() for a in k0),
            head + b"".join(a.tobytes() for a in k1),
        )

    def _orbit_key(self, f, ivs):
        """Key invariant under the reparametrisations ``b_j -> +-b_j^(+-1)``
        of each slot's generic value and under permutations of the values
        over the slots (each is a bijection of every slot's interval)."""
        k = len(ivs)
        base = _POOL_REAL[:k]
        perms = np.array(list(itertools.permutations(range(k))))
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=k)))
        powers = np.array(list(itertools.product([1.0, -1.0], repeat=k)))
        n_tr = len(signs) * len(powers)
        if n_tr * len(perms) > _MAX_GROUP:
            perms = perms[: max(1, _MAX_GROUP // n_tr)]
        b = base[perms]  # (P, k)
        b = (
            np.power(b[:, None, None, :], powers[None, None, :, :])
            * signs[None, :, None, :]
        ).reshape(-1, k)
        th = np.column_stack([_phi(iv, b[:, j]) for j, iv in enumerate(ivs)])
        th = np.unique(th, axis=0)
        vals = _family_eval(f, th)
        # The whole set of values over the orbit (not e.g. its smallest
        # member), so that a family is not confused with a larger one that
        # contains it, such as x^x * c for c >= 0 and for all real c
        out = []
        for q in _quantise(vals, _REL_STEP, _ZERO):
            rows = sorted(set(_rows_to_bytes(q)))
            out.append(b"O" + hashlib.blake2b(b"".join(rows), digest_size=20).digest())
        return tuple(out)


def _batched_least_squares(
    f, ivs, fin, U, Y, transform=None, n_iter=200, tol=None, group=1
):
    """Minimise ``|T(F(theta(u))) - y|`` for many problems at once.

    Damped Gauss-Newton (Levenberg-Marquardt) on the unconstrained
    coordinates ``u`` of the slots (see ``_to_slot``), one problem per row of
    ``U`` and ``Y``. Each iteration evaluates ``f`` once for all problems.
    If ``tol`` is given, stops once every consecutive block of ``group``
    problems (the starting points for one target) has one problem with
    residual below ``tol``, or once no problem above ``tol`` is improving.

    Returns:
        :res (numpy.ndarray): the residual norm reached for each problem
        :U (numpy.ndarray): the coordinates reached for each problem
    """
    k = U.shape[1]
    T = transform if transform is not None else (lambda v: v)

    def evaluate(u):
        th = np.column_stack([_to_slot(iv, u[:, j]) for j, iv in enumerate(ivs)])
        return T(_family_eval(f, th)[:, fin])

    def norms(r):
        n = np.linalg.norm(r, axis=1)
        return np.where(np.isfinite(n), n, np.inf)

    R = evaluate(U) - Y
    cost = norms(R)
    lam = np.full(len(U), 1e-3)
    eye = np.eye(k)
    last = cost.copy()
    for it in range(n_iter):
        h = 1e-7 * np.maximum(np.abs(U), 1e-12)
        pert = np.repeat(U, k, axis=0) + (h[:, :, None] * eye).reshape(-1, k)
        Fp = evaluate(pert).reshape(len(U), k, -1)
        J = (Fp - (R + Y)[:, None, :]) / h[:, :, None]  # (P, k, n)
        J = np.where(np.isfinite(J), J, 0.0)
        Rf = np.where(np.isfinite(R), R, 0.0)
        A = J @ J.transpose(0, 2, 1)
        g = (J @ Rf[:, :, None])[:, :, 0]
        diag = np.einsum("pii->pi", A)
        A = A + lam[:, None, None] * (diag[:, :, None] * eye + 1e-12 * eye)
        try:
            step = -np.linalg.solve(A, g[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            # A singular system must not stop the other problems in the batch
            step = -np.einsum("pij,pj->pi", np.linalg.pinv(A), g)
        step = np.where(np.isfinite(step), step, 0.0)
        Unew = U + step
        Rnew = evaluate(Unew) - Y
        cnew = norms(Rnew)
        better = cnew < cost
        U = np.where(better[:, None], Unew, U)
        R = np.where(better[:, None], Rnew, R)
        cost = np.where(better, cnew, cost)
        lam = np.where(better, lam / 3, lam * 4)
        lam = np.clip(lam, 1e-12, 1e12)
        if tol is not None and (it % 10 == 9):
            done = cost < tol
            if done.reshape(-1, group).any(axis=1).all():
                break
            stalled = (cost >= (1 - 1e-6) * last) & (lam >= 1e3)
            if np.all(done | stalled):
                break
            last = cost.copy()
    return cost, U


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def _variable_points(name):
    rs = np.random.RandomState(zlib.crc32(("unique_enumerator:" + name).encode()))
    return rs.uniform(0.2, 5.0, N_POINTS)


class Enumerator:
    """Enumerates the distinct functions of a basis, by increasing complexity.

    Args:
        :basis_functions (list): ``[nullary, unary, binary]`` basis functions.
            Nullary entries are variables, the parameter ``"a"`` or numbers.
        :variables (list): names of the variables in the nullary basis

    Attributes:
        :levels (dict): ``levels[c]`` is the list of ``Func`` of complexity ``c``
    """

    def __init__(self, basis_functions, variables):
        nullary, unary, binary = basis_functions
        for op in unary:
            if op not in UNARY:
                raise ValueError(f"Unary operator {op!r} is not supported")
        for op in binary:
            if op not in BINARY:
                raise ValueError(f"Binary operator {op!r} is not supported")
        self.nullary = list(nullary)
        self.unary = list(unary)
        self.binary = list(binary)
        self.var_points = {v: _variable_points(v) for v in variables}
        self.keyer = _Keyer()
        self.levels = {}
        self._seen = ({}, {})
        self.n_candidates = {}

    def _offer(self, f, c):
        """Keep ``f`` if it is new. If it is the same as a function already
        kept at this complexity, keep whichever tree has the lower ``aifeyn``
        contribution, preferring trees affine in their parameters (which are
        easier to fit) and then shorter strings."""
        if f is None:
            return
        keys = self.keyer.key(f)
        if keys is None:
            return
        old = None
        for key in keys:
            old = self._seen[0].get(key[0]) or self._seen[1].get(key[1])
            if old is not None:
                break
        holder = f if old is None else old
        for key in keys:
            self._seen[0].setdefault(key[0], holder)
            self._seen[1].setdefault(key[1], holder)
        if old is None:
            self.levels[c].append(f)
            return
        if old.c == c:
            new_rank = (f.aifeyn(), not f.linear, len(f.string()), f.string())
            old_rank = (old.aifeyn(), not old.linear, len(old.string()), old.string())
            if new_rank < old_rank:
                for attr in Func.__slots__:
                    setattr(old, attr, getattr(f, attr))

    def run(self, compl, verbose=True):
        """Enumerate all functions up to complexity ``compl``."""
        for c in range(1, compl + 1):
            if c in self.levels:
                continue
            self.levels[c] = []
            n = 0
            if c == 1:
                for label in self.nullary:
                    n += 1
                    self._offer(_leaf(label, self.var_points), c)
            else:
                for op in self.unary:
                    for g in self.levels[c - 1]:
                        n += 1
                        self._offer(_combine(op, [g]), c)
                for op in self.binary:
                    for i in range(1, c - 1):
                        j = c - 1 - i
                        if op in COMMUTATIVE and i > j:
                            continue
                        for ia, fa in enumerate(self.levels[i]):
                            start = ia if (op in COMMUTATIVE and i == j) else 0
                            for fb in self.levels[j][start:]:
                                n += 1
                                self._offer(_combine(op, [fa, fb]), c)
            self.n_candidates[c] = n
            if verbose:
                print(
                    f"Complexity {c}: {len(self.levels[c])} new functions "
                    f"from {n} candidates",
                    flush=True,
                )
        return self.levels


def write_catalogue(levels, dirname):
    """Write one ``compl_<c>`` directory per complexity in the format read by
    the fitting routines."""
    for c, funcs in sorted(levels.items()):
        d = os.path.join(dirname, f"compl_{c}")
        os.makedirs(d, exist_ok=True)
        strings = [f.string() for f in funcs]
        n = len(strings)
        for name in ["all_equations", "unique_equations"]:
            with utils.atomic_write(os.path.join(d, f"{name}_{c}.txt")) as fh:
                fh.write("".join(s + "\n" for s in strings))
        with utils.atomic_write(os.path.join(d, f"matches_{c}.txt")) as fh:
            fh.write("".join(f"{i}\n" for i in range(n)))
        with utils.atomic_write(os.path.join(d, f"inv_subs_{c}.txt")) as fh:
            fh.write("\n" * n)
        with utils.atomic_write(os.path.join(d, f"aifeyn_{c}.txt")) as fh:
            fh.write("".join(f"{f.aifeyn()}\n" for f in funcs))
        with utils.atomic_write(os.path.join(d, f"trees_{c}.txt")) as fh:
            fh.write("".join(f"{f.prefix()}\n" for f in funcs))


def main(
    runname, compl, variables=None, basis_functions=None, fn_dir=None, verbose=True
):
    """Enumerate the distinct functions up to a given complexity and save them.

    The arguments have the same meaning as for ``duplicate_checker.main``,
    except that every complexity from 1 to ``compl`` is generated at once
    (each one is needed to build the next).

    Args:
        :runname (str): name of run, which defines the basis functions used
        :compl (int): maximum complexity of functions to consider
        :variables (list, default=None): names of the input variables
        :basis_functions (list, default=None): custom basis functions
            ``[nullary, unary, binary]`` to use instead of those of ``runname``
        :fn_dir (str, default=None): output directory; by default the
            directory a likelihood with ``fn_set=runname`` reads from
        :verbose (bool, default=True): whether to print progress

    Returns:
        :levels (dict): ``levels[c]`` is the list of ``Func`` of complexity
            ``c`` (on rank 0; None on other ranks)
    """
    custom_basis = basis_functions is not None
    if custom_basis:
        basis_functions = [list(b) for b in basis_functions]
    else:
        basis_functions = builtin_basis_functions(runname)
    basis_functions, variables = _apply_variables(
        basis_functions, variables, custom=custom_basis
    )
    register_variables(variables)

    if fn_dir is None:
        dirname = os.path.abspath(
            os.path.join(os.path.dirname(generator.__file__), "..", "function_library")
        )
        dirname = os.path.join(dirname, function_set_dirname(runname, variables))
    else:
        dirname = os.path.abspath(fn_dir)

    levels = None
    if rank == 0:
        if verbose:
            print("Basis functions:", basis_functions, flush=True)
        levels = Enumerator(basis_functions, variables).run(compl, verbose=verbose)
        write_catalogue(levels, dirname)
        if verbose:
            total = sum(len(v) for v in levels.values())
            print(f"Saved {total} functions to {dirname}", flush=True)
        sys.stdout.flush()
    comm.Barrier()
    return levels

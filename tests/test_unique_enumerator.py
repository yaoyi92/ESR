"""Tests for the duplicate-free enumerator ``esr.generation.unique_enumerator``."""

import os
import re

import numpy as np
import pytest
import sympy

import esr.fitting.combine_dl
import esr.fitting.match
import esr.fitting.test_all
import esr.fitting.test_all_fisher
from esr.fitting.likelihood import GaussLikelihood
from esr.fitting.sympy_symbols import make_variables, sympy_locs
from esr.generation import unique_enumerator as ue
from esr.generation.duplicate_checker import builtin_basis_functions

VP = {"x": ue._variable_points("x")}
X = ue._leaf("x", VP)
A = ue._leaf("a", VP)


def C(op, *kids):
    return ue._combine(op, list(kids))


def keys_match(f, g):
    keyer = ue._Keyer()
    kf, kg = keyer.key(f), keyer.key(g)
    assert kf is not None and kg is not None
    return any(a[0] == b[0] or a[1] == b[1] for a in kf for b in kg)


def test_intervals_of_constants():
    assert C("+", A, A).iv == ue.REAL
    assert C("inv", A).iv == ue.REAL
    assert C("pow", A, A).iv == (0.0, np.inf)
    assert C("sqrt_abs", A).iv == (0.0, np.inf)
    assert C("exp", A).iv == (0.0, np.inf)
    assert C("sin", A).iv == (-1.0, 1.0)
    assert C("+", C("exp", A), ue._leaf("1", VP)).iv == (1.0, np.inf)
    # x/x is the number 1, so 1 - c (c >= 0) is a constant in (-inf, 1]
    one = C("/", X, X)
    assert not one.has_var
    assert C("-", one, C("pow", A, A)).iv == (-np.inf, 1.0)


@pytest.mark.parametrize(
    "f, g",
    [
        # affine in the parameters, up to reparametrisation
        (C("*", A, C("+", X, A)), C("+", C("*", X, A), A)),
        (C("inv", C("/", A, C("-", A, X))), C("-", C("*", X, A), A)),
        # affine after taking log|f|
        (C("/", C("pow", A, X), A), C("*", A, C("pow", A, X))),
        (C("pow", C("*", X, A), A), C("pow", C("/", A, X), A)),
        # a0 -> -a0 and permutations of the parameters
        (C("pow", C("+", X, A), A), C("pow", C("-", A, X), A)),
        # |a0|^x with a real a0 and c^x with c >= 0
        (C("pow", A, X), C("pow", C("pow", A, A), X)),
    ],
)
def test_equal_families_have_equal_keys(f, g):
    assert keys_match(f, g)


@pytest.mark.parametrize(
    "f, g",
    [
        (A, C("pow", A, A)),  # any real vs non-negative constant
        (C("+", X, A), C("+", X, C("pow", A, A))),
        (C("*", A, C("pow", A, X)), C("pow", A, C("+", X, A))),  # C b^x vs C > 0
        (C("+", X, A), C("-", A, X)),
    ],
)
def test_different_families_have_different_keys(f, g):
    assert not keys_match(f, g)


def test_rank_deficient_families_are_dropped():
    # exp-like a0^(a1*x) = (a0^a1)^x has one independent direction
    f = C("pow", A, C("*", A, X))
    keyer = ue._Keyer()
    keys = keyer.key(f)
    # it is the same family as pow(a0, x) (via its log-affine key), or dropped
    assert keys is None or keys_match(f, C("pow", A, X))


def test_arithmetic_three_variables():
    """Expressions in three variables using each once, with + - * /: 68 of them
    (cf. github.com/yaoyi92/enumerate-expressions), and no two equal."""
    variables = ["x", "y", "z"]
    E = ue.Enumerator([variables, [], ["+", "-", "*", "/"]], variables)
    levels = E.run(5, verbose=False)
    funcs = [f.string() for c in levels for f in levels[c]]
    once = [
        s for s in levels[5] if sorted(re.findall(r"[a-z]", s.string())) == variables
    ]
    assert len(once) == 68

    syms = make_variables(variables)
    loc = dict(zip(variables, syms))
    points = [
        dict(zip(syms, map(sympy.Integer, p)))
        for p in [(3141592, 6535897, 9323846), (271828, 161803, 141421)]
    ]
    values = {}
    for s in funcs:
        e = sympy.sympify(s, locals=loc)
        values.setdefault(tuple(str(e.subs(p)) for p in points), []).append(s)
    assert [v for v in values.values() if len(v) > 1] == []


def test_core_maths_is_duplicate_free_and_parsable():
    E = ue.Enumerator(builtin_basis_functions("core_maths"), ["x"])
    levels = E.run(4, verbose=False)
    strings = [f.string() for c in levels for f in levels[c]]
    assert len(strings) == len(set(strings))
    assert "x*a0" in strings and "pow(a0, a1)" in strings
    for s in strings:
        sympy.sympify(s, locals=sympy_locs)
    # each function is listed only at its lowest complexity
    assert "a0" in [f.string() for f in levels[1]]
    assert all("a0+a1" != f.string() for f in levels[3])


def test_main_writes_catalogue_and_fits(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.uniform(0.5, 3, 60)
    y = 2 * x + 1 + rng.normal(0, 0.1, 60)
    np.savetxt(tmp_path / "data.txt", np.c_[x, y, np.full(60, 0.1)])
    likelihood = GaussLikelihood(
        "data.txt",
        "unique",
        data_dir=str(tmp_path),
        fn_dir=str(tmp_path / "functions"),
        base_out_dir=str(tmp_path / "out"),
    )
    levels = ue.main("core_maths", 3, fn_dir=likelihood.fn_dir, verbose=False)

    for c in range(1, 4):
        d = os.path.join(likelihood.fn_dir, f"compl_{c}")
        lines = {}
        for name in ["all_equations", "unique_equations", "matches", "aifeyn"]:
            with open(os.path.join(d, f"{name}_{c}.txt")) as f:
                lines[name] = f.read().splitlines()
        n = len(levels[c])
        assert all(len(v) == n for v in lines.values())
        assert lines["all_equations"] == lines["unique_equations"]
        assert lines["matches"] == [str(i) for i in range(n)]

    for c in range(1, 4):
        esr.fitting.test_all.main(c, likelihood, Niter_params=[4], Nconv_params=[2])
        esr.fitting.test_all_fisher.main(c, likelihood)
        esr.fitting.match.main(c, likelihood)
        esr.fitting.combine_dl.main(c, likelihood)
    with open(os.path.join(likelihood.out_dir, "final_3.dat")) as f:
        best = f.readline().split(";")[1]
    assert best == "x*a0"

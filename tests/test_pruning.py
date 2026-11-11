from collections import Counter

import numpy as np
import pytest

import esr.generation.duplicate_checker
from esr.generation import generator, pruning

BASIS = [["x", "a"], ["inv"], ["+", "*", "-", "/", "pow"]]
UNARY, BINARY = set(BASIS[1]), set(BASIS[2])


def _nested(labels):
    return pruning.labels_to_nested([str(lab) for lab in labels], UNARY, BINARY)


def _generate(compl, prune):
    """All trees the generator produces at this complexity, as nested tuples"""
    trees = []
    for s in generator.get_allowed_shapes(compl):
        _, shape_trees, _, _, _ = generator.shape_to_functions(s, BASIS, prune=prune)
        trees += [_nested(t) for t in shape_trees]
    return trees


def _reorder_key(t):
    """Identical for trees differing only in the order of operands of + and *"""
    if len(t) == 1:
        return t
    if len(t) == 2:
        return (t[0], _reorder_key(t[1]))
    left, right = _reorder_key(t[1]), _reorder_key(t[2])
    if t[0] in ("+", "*"):
        operands = []
        for c in (left, right):
            operands += list(c[1]) if c[0] == "chain" + t[0] else [c]
        return ("chain" + t[0], tuple(sorted(operands)))
    return (t[0], left, right)


@pytest.mark.parametrize(
    "labels, redundant",
    [
        # Reordering: one ordering of each sum / product survives
        (["+", "a0", "x"], False),
        (["+", "x", "a0"], True),
        (["*", "a0", "x"], False),
        (["*", "x", "a0"], True),
        (["+", "x", "+", "x", "x"], False),
        (["+", "+", "x", "x", "x"], True),
        # ... but non-commutative operators are left alone
        (["-", "x", "a0"], False),
        (["-", "a0", "x"], False),
        (["/", "x", "a0"], False),
        (["/", "a0", "x"], False),
        (["pow", "x", "a0"], False),
        (["pow", "a0", "x"], False),
        # A subtree of parameters only is just a parameter
        (["a0"], False),
        (["inv", "a0"], True),
        (["+", "a0", "a1"], True),
        (["*", "x", "pow", "a0", "a1"], True),
        # inv(inv(f)) = f
        (["inv", "x"], False),
        (["inv", "inv", "x"], True),
        (["inv", "+", "a0", "inv", "x"], False),
        # Two parameters in one sum
        (["-", "-", "x", "a0", "a1"], True),
        (["-", "a0", "-", "x", "a1"], True),
        (["+", "*", "a0", "x", "a1"], False),
        (["+", "a0", "*", "a1", "x"], True),
        # A parameter scaling something which already absorbs a scale
        (["/", "/", "x", "a0", "a1"], True),
        (["/", "a0", "*", "a1", "x"], True),
        (["*", "+", "*", "a0", "x", "a1", "a2"], True),
        (["*", "+", "a0", "x", "a1"], False),
        (["/", "a0", "+", "*", "a1", "x", "a2"], False),
        (["*", "a0", "pow", "x", "a1"], False),
    ],
)
def test_rules(labels, redundant):
    assert pruning.make_filter(BASIS)(labels) is redundant


def test_filter_applies_to_other_basis_sets():
    basis = [["x", "a"], ["inv", "exp", "square"], ["+", "*", "-", "/", "pow"]]
    is_redundant = pruning.make_filter(basis)
    assert is_redundant(["exp", "a0"])
    assert is_redundant(["+", "exp", "x", "a0"])
    assert not is_redundant(["+", "a0", "exp", "x"])
    assert not is_redundant(["exp", "exp", "x"])
    assert not is_redundant(["*", "a0", "exp", "*", "a1", "x"])
    with pytest.raises(ValueError):
        is_redundant(["+", "x", "x", "x"])


@pytest.mark.parametrize("compl", [1, 2, 3, 4, 5, 6])
def test_pruned_generation_matches_filtered_full_generation(compl):
    full = _generate(compl, prune=False)
    pruned = _generate(compl, prune=True)

    # The full generation is unchanged and has no repeated tree
    expected_full = {1: 2, 2: 2, 3: 22, 4: 62, 5: 522, 6: 2202}[compl]
    assert len(full) == len(set(full)) == expected_full

    # Pruned generation gives exactly the trees of the full generation which
    # pass the filter, in the same order
    assert pruned == [t for t in full if not pruning.nested_is_redundant(t)]
    assert len(pruned) == {1: 2, 2: 1, 3: 13, 4: 29, 5: 209, 6: 731}[compl]

    # No two surviving trees differ only by the order of their operands
    keys = Counter(_reorder_key(t) for t in pruned)
    assert max(keys.values()) == 1


def test_reordering_alone_never_removes_a_function():
    """Every tree rejected for the order of its operands has a reordering
    which is not rejected for that reason"""

    def ordered(t):
        # Apply only the reordering rule at every node
        if len(t) == 1:
            return True
        if len(t) == 3 and t[0] in ("+", "*"):
            if len(t[1]) == 3 and t[1][0] == t[0]:
                return False
            nxt = t[2][1] if (len(t[2]) == 3 and t[2][0] == t[0]) else t[2]
            if t[1] > nxt:
                return False
        return all(ordered(c) for c in t[1:])

    for compl in range(1, 7):
        groups = {}
        for t in _generate(compl, prune=False):
            groups.setdefault(_reorder_key(t), []).append(t)
        for group in groups.values():
            assert sum(ordered(t) for t in group) == 1


@pytest.mark.parametrize(
    "pruned, kept, convert",
    [
        # (rejected tree, smaller equivalent tree, parameters of the latter)
        ("a0 + a1", "a0", lambda a: [a[0] + a[1]]),
        ("x * abs(a0) ** a1", "a0 * x", lambda a: [abs(a[0]) ** a[1]]),
        ("1 / (1 / (a0 + x))", "a0 + x", lambda a: [a[0]]),
        ("(x - a0) - a1", "x - a0", lambda a: [a[0] + a[1]]),
        ("a0 - (x - a1)", "a0 - x", lambda a: [a[0] + a[1]]),
        ("(x / a0) / a1", "x / a0", lambda a: [a[0] * a[1]]),
        ("a0 / (a1 * x)", "a0 / x", lambda a: [a[0] / a[1]]),
        (
            "a0 * (a1 + a2 * x)",
            "a0 + a1 * x",
            lambda a: [a[0] * a[1], a[0] * a[2]],
        ),
        (
            "a0 / (a1 / x + a2 * x)",
            "1 / (a0 / x + a1 * x)",
            lambda a: [a[1] / a[0], a[2] / a[0]],
        ),
    ],
)
def test_dominated_trees_are_reproduced_by_smaller_ones(pruned, kept, convert):
    x = np.linspace(0.3, 3.0, 25)
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.uniform(0.5, 2.0, 3) * rng.choice([-1, 1], 3)
        assert np.allclose(
            eval(pruned, {"x": x, "a0": a[0], "a1": a[1], "a2": a[2]}) + 0 * x,
            eval(kept, dict(zip(["a0", "a1"], convert(a)), x=x)) + 0 * x,
        )


def test_extra_trees_are_kept_once():
    seen = set()
    extra = []
    for s in generator.get_allowed_shapes(5):
        _, _, _, extra_tree, _ = generator.shape_to_functions(
            s, BASIS, prune=True, seen_extra=seen
        )
        extra += [tuple(str(lab) for lab in t) for t in extra_tree]
    assert len(extra) > 0
    assert len(extra) == len(set(extra)) == len(seen)


def _extra_keys(compl, prune):
    """Order-independent keys of the extra trees found at this complexity, and
    the originals they refer to"""
    _, canonical_key, _ = pruning.make_classifier(BASIS)
    keys, orig, kept = set(), set(), set()
    for s in generator.get_allowed_shapes(compl):
        all_fun, _, _, extra_tree, extra_orig = generator.shape_to_functions(
            s, BASIS, prune=prune
        )
        keys |= {canonical_key([str(lab) for lab in t]) for t in extra_tree}
        orig |= set(extra_orig)
        kept |= set(all_fun)
    return keys, orig, kept


@pytest.mark.parametrize("compl", [3, 4, 5, 6])
def test_pruning_keeps_extra_trees_of_reordered_trees(compl):
    # The search for extra trees depends on the order of the operands, so the
    # reorderings which are not kept have to be searched as well
    full, _, _ = _extra_keys(compl, False)
    pruned, orig, kept = _extra_keys(compl, True)
    assert pruned <= full
    # Only extra trees of dominated originals may go
    for key in full - pruned:
        assert pruning.classify_nested(key) == pruning.DOMINATED, key
    # Every extra tree refers to an original which was kept
    assert orig <= kept


def test_extra_tree_search_is_order_dependent():
    # x + x is only combined to 2 * x in the second ordering, which is not the
    # one kept by the pruning; the kept ordering is used as the original
    reordered = ["*", "pow", "+", "x", "x", "x", "-", "a0", "x"]
    kept = ["*", "-", "a0", "x", "pow", "+", "x", "x", "x"]
    classify, canonical_key, canonical_labels = pruning.make_classifier(BASIS)
    assert classify(reordered) == pruning.REORDERED
    assert classify(kept) == pruning.KEEP
    assert canonical_labels(reordered) == (kept, {"a0": "a0"})
    assert canonical_key(reordered) == canonical_key(kept)

    found = {}
    for labels in [reordered, kept]:
        shape = np.array(generator.labels_to_shape(labels, BASIS))
        _, _, tree = generator.check_tree(shape)
        _, new_labels = generator.find_additional_trees(tree, list(labels), BASIS)
        found[tuple(labels)] = new_labels[1:]
    assert found[tuple(kept)] == []
    assert found[tuple(reordered)] == [["*", "pow", "*", "2", "x", "x", "-", "a0", "x"]]


def test_canonical_labels_rename_parameters():
    _, _, canonical_labels = pruning.make_classifier(BASIS)
    new, rename = canonical_labels(["+", "pow", "x", "a0", "*", "a1", "x"])
    assert new == ["+", "*", "a0", "x", "pow", "x", "a1"]
    assert rename == {"a1": "a0", "a0": "a1"}


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["+", "x", "a0"], ["+", "a0", "x"]),
        (["*", "x", "+", "x", "a0"], ["*", "+", "a0", "x", "x"]),
        (["+", "+", "x", "a0", "inv", "x"], ["+", "a0", "+", "inv", "x", "x"]),
        (["-", "*", "x", "a0", "+", "x", "a1"], ["-", "*", "a0", "x", "+", "a1", "x"]),
        (["pow", "x", "a0"], ["pow", "x", "a0"]),
        # Compared as binary trees, not as flattened chains
        (
            ["*", "+", "x", "+", "x", "x", "+", "x", "x"],
            ["*", "+", "x", "+", "x", "x", "+", "x", "x"],
        ),
        (
            ["*", "+", "x", "x", "+", "+", "x", "x", "x"],
            ["*", "+", "x", "+", "x", "x", "+", "x", "x"],
        ),
    ],
)
def test_canonical_labels(labels, expected):
    classify, _, canonical_labels = pruning.make_classifier(BASIS)
    new, rename = canonical_labels(labels)
    assert new == expected
    assert sorted(rename) == sorted(rename.values())
    assert classify(expected) == pruning.KEEP


def _line_count(fname):
    with open(fname, "r") as f:
        return sum(1 for _ in f)


def test_generation_with_pruning(tmp_path):
    """The whole generation runs on the pruned trees, including a complexity
    where no surviving function has a parameter"""
    nunique = {}
    for compl in range(1, 6):
        esr.generation.duplicate_checker.main(
            "core_maths", compl, fn_dir=str(tmp_path), prune_redundant=True
        )
        d = tmp_path / f"compl_{compl}"
        ntree = _line_count(d / f"trees_{compl}.txt")
        for name in ["all_equations", "matches", "aifeyn", "inv_subs"]:
            assert _line_count(d / f"{name}_{compl}.txt") == ntree
        nunique[compl] = _line_count(d / f"unique_equations_{compl}.txt")
        matches = np.atleast_1d(np.loadtxt(d / f"matches_{compl}.txt")).astype(int)
        assert matches.min() >= 0 and matches.max() < nunique[compl]

    with open(tmp_path / "compl_2" / "unique_equations_2.txt", "r") as f:
        assert f.read().split() == ["1/x"]
    assert nunique[1] == 2
    assert nunique[5] < 131  # fewer than without pruning


def test_parameters_are_anonymous_when_ordering_operands():
    assert _nested(["+", "a0", "a1"]) == _nested(["+", "a1", "a0"])
    assert _nested(["*", "a0", "x"]) < _nested(["*", "x", "a0"])

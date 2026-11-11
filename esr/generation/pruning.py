"""Generation-time pruning of redundant trees.

The exhaustive generator labels every tree topology with every combination of
basis functions, so most of the trees it produces are repeats of one another.
The rules here reject a tree *before* it is turned into a string and handed to
sympy, but only when an equivalent tree that is never worse for the description
length is certain to be generated as well:

* **Reordering** (commutativity and associativity of ``+`` and ``*``): exactly
  one ordering of each sum / product chain is kept. The rejected trees have the
  same multiset of labels as the kept one, hence the same tree term
  ``k log(n)`` and the same fitted parameters.
* **Dominance**: the tree describes the same family of functions as a strictly
  smaller tree built from a subset of its labels, so the smaller tree (found at
  a lower complexity) has a tree term which is no larger, and no more
  parameters:

  - a subtree containing only parameters is itself just a parameter
    (``a0 + a1``, ``inv(a0)``, ``pow(a0, a1)`` ...);
  - ``inv(inv(f))`` is ``f``;
  - two bare parameters in one ``+``/``-`` chain collapse to one
    (``a0 + x - a1``);
  - a bare parameter multiplying or dividing something which can already
    absorb an overall scale is superfluous (``a0 * (a1 * x)``,
    ``a0 * (a1 + a2 * x)``).

Because the dominance rules rely on the smaller tree appearing at a lower
complexity, a pruned catalogue at complexity ``k`` is only complete when used
together with the catalogues for all complexities below ``k``.

The generator's search for extra trees with integer coefficients (``x + x``
to ``2 * x``) depends on the order of the operands, so trees rejected only for
their ordering are still searched for extra trees; the extra trees found are
then kept once per ordering (see ``classify_nested`` and ``canonical_nested``).

Rules which swap one operator for another (``f - a`` vs ``f + a``,
``f / a`` vs ``a * f``, removing brackets) are deliberately *not* applied: they
can change the tree term or the parameter codelength of the best variant.
"""

_PARAM = ("a",)


def is_param(label):
    """Whether a node label is a free parameter (``a`` or ``a<integer>``)"""
    return label[:1] == "a" and (len(label) == 1 or label[1:].isdigit())


def labels_to_nested(labels, unary, binary, anonymise=True):
    """Convert a prefix-ordered list of node labels to a nested tuple.

    Leaves are ``(name,)`` with all parameters anonymised to ``("a",)``, unary
    nodes are ``(op, child)`` and binary nodes are ``(op, left, right)``.

    Args:
        :labels (list): list of strings giving node labels of tree, in the
            (prefix) order used throughout the generator
        :unary (set): labels of the unary operators
        :binary (set): labels of the binary operators
        :anonymise (bool, default=True): whether to replace the names of the
            parameters by ``a``

    Returns:
        :tree (tuple): nested tuple representation of the tree
    """
    pos = 0

    def build():
        nonlocal pos
        t = labels[pos]
        pos += 1
        if t in binary:
            left = build()
            right = build()
            return (t, left, right)
        if t in unary:
            return (t, build())
        return _PARAM if (anonymise and is_param(t)) else (t,)

    tree = build()
    if pos != len(labels):
        raise ValueError(f"Labels do not form a single tree: {labels}")
    return tree


def _has_variable(t):
    """Whether the (sub)tree contains a leaf which is not a free parameter"""
    if len(t) == 1:
        return t != _PARAM
    return any(_has_variable(c) for c in t[1:])


def _chain(t, plus, minus):
    """Flatten a chain of ``plus``/``minus`` nodes.

    Returns the operands entering with a positive sign (terms added, or factors
    in the numerator) and those entering with a negative sign (terms
    subtracted, or factors in the denominator).
    """
    if len(t) == 3 and t[0] == plus:
        lp, ln = _chain(t[1], plus, minus)
        rp, rn = _chain(t[2], plus, minus)
        return lp + rp, ln + rn
    if len(t) == 3 and t[0] == minus:
        lp, ln = _chain(t[1], plus, minus)
        rp, rn = _chain(t[2], plus, minus)
        return lp + rn, ln + rp
    return [t], []


def _absorbs_scale(t):
    """Whether multiplying the subtree by an arbitrary constant gives a
    function which the subtree can already represent, by a change of its own
    parameters"""
    if t == _PARAM:
        return True
    op = t[0]
    if len(t) == 3 and op in ("*", "/"):
        num, den = _chain(t, "*", "/")
        return any(_absorbs_scale(f) for f in num + den)
    if len(t) == 3 and op in ("+", "-"):
        pos, neg = _chain(t, "+", "-")
        return all(_absorbs_scale(f) for f in pos + neg)
    if len(t) == 2 and op == "inv":
        return _absorbs_scale(t[1])
    return False


def _node_misordered(s):
    """Whether the root of the subtree ``s`` is a ``+`` or ``*`` chain which
    is not in canonical form (nested to the right, operands in non-decreasing
    order)"""
    if len(s) != 3 or s[0] not in ("+", "*"):
        return False
    op, left, right = s
    if len(left) == 3 and left[0] == op:
        return True
    nxt = right[1] if (len(right) == 3 and right[0] == op) else right
    return left > nxt


def _node_dominated(s):
    """Apply the dominance rules at the root of the subtree ``s``"""
    if len(s) == 1:
        return False

    # A parameter-only subtree is just a parameter
    if not _has_variable(s):
        return True

    op = s[0]

    if len(s) == 2:
        # inv(inv(f)) = f
        return op == "inv" and s[1][0] == "inv" and len(s[1]) == 2

    if op in ("+", "-"):
        # Two bare parameters in one sum are a single parameter
        pos, neg = _chain(s, "+", "-")
        if pos.count(_PARAM) + neg.count(_PARAM) >= 2:
            return True

    if op in ("*", "/"):
        # A bare parameter multiplying something which already absorbs a
        # scale. Removing a parameter from the numerator must leave something
        # there, as otherwise we would need an extra inv.
        num, den = _chain(s, "*", "/")
        removable = _PARAM in den or (_PARAM in num and len(num) >= 2)
        if removable and sum(_absorbs_scale(f) for f in num + den) >= 2:
            return True

    return False


KEEP, REORDERED, DOMINATED = 0, 1, 2


def classify_nested(tree):
    """Classify a tree in nested tuple form

    Args:
        :tree (tuple): nested tuple representation, see labels_to_nested

    Returns:
        :status (int): DOMINATED if a strictly smaller tree describes the same
            functions, otherwise REORDERED if the tree is a non-canonical
            ordering of a sum or product, otherwise KEEP
    """
    status = KEEP
    stack = [tree]
    while stack:
        s = stack.pop()
        if _node_dominated(s):
            return DOMINATED
        if status == KEEP and _node_misordered(s):
            status = REORDERED
        stack.extend(s[1:])
    return status


def nested_is_redundant(tree):
    """Whether a tree in nested tuple form is rejected by any pruning rule

    Args:
        :tree (tuple): nested tuple representation, see labels_to_nested

    Returns:
        :redundant (bool): True if an equivalent tree which is never worse is
            generated elsewhere, so that this one can be skipped
    """
    return classify_nested(tree) != KEEP


def _anonymous(t):
    """Copy of a tree in nested tuple form with all parameters named ``a``"""
    if len(t) == 1:
        return _PARAM if is_param(t[0]) else t
    return (t[0], *(_anonymous(c) for c in t[1:]))


def canonical_nested(t):
    """The reordering of a tree in nested tuple form which is not REORDERED:
    every ``+`` and ``*`` chain is nested to the right with its operands
    sorted (ignoring the names of the parameters). Two trees which differ only
    by reordering sums and products give the same result, up to the names of
    the parameters."""
    if len(t) == 1:
        return t
    if len(t) == 2:
        return (t[0], canonical_nested(t[1]))
    op = t[0]
    if op not in ("+", "*"):
        return (op, canonical_nested(t[1]), canonical_nested(t[2]))
    operands, _ = _chain(t, op, None)
    operands = sorted((canonical_nested(c) for c in operands), key=_anonymous)
    out = operands[-1]
    for c in reversed(operands[:-1]):
        out = (op, c, out)
    return out


def nested_to_labels(t):
    """Prefix-ordered node labels of a tree in nested tuple form"""
    labels = []
    stack = [t]
    while stack:
        s = stack.pop()
        labels.append(s[0])
        stack.extend(reversed(s[1:]))
    return labels


def make_filter(basis_functions):
    """Make a function which says whether a labelled tree can be skipped

    Args:
        :basis_functions (list): list of lists basis functions.
            basis_functions[0] are nullary, basis_functions[1] are unary and
            basis_functions[2] are binary operators

    Returns:
        :is_redundant (callable): takes the list of node labels of a tree (in
            prefix order) and returns True if the tree is redundant
    """
    unary = set(basis_functions[1])
    binary = set(basis_functions[2])

    def is_redundant(labels):
        return nested_is_redundant(labels_to_nested(labels, unary, binary))

    return is_redundant


def make_classifier(basis_functions):
    """Make a function which classifies a labelled tree as KEEP, REORDERED or
    DOMINATED (see classify_nested), and one which gives a key shared by all
    reorderings of a labelled tree, and one which gives the reordering which
    is kept

    Args:
        :basis_functions (list): list of lists basis functions.
            basis_functions[0] are nullary, basis_functions[1] are unary and
            basis_functions[2] are binary operators

    Returns:
        :classify (callable): takes the list of node labels of a tree (in
            prefix order) and returns its status
        :canonical_key (callable): takes the list of node labels of a tree and
            returns a hashable key which ignores the order of sums and products
        :canonical_labels (callable): takes the list of node labels of a tree
            and returns the labels of the reordering classified as KEEP, and
            a dictionary from the names of the parameters in the tree to their
            names in that reordering
    """
    unary = set(basis_functions[1])
    binary = set(basis_functions[2])

    def classify(labels):
        return classify_nested(labels_to_nested(labels, unary, binary))

    def canonical_key(labels):
        return canonical_nested(labels_to_nested(labels, unary, binary))

    def canonical_labels(labels):
        named = labels_to_nested(labels, unary, binary, anonymise=False)
        new = nested_to_labels(canonical_nested(named))
        rename = {}
        for i, lab in enumerate(new):
            if is_param(lab):
                rename[lab] = f"a{len(rename)}"
                new[i] = rename[lab]
        return new, rename

    return classify, canonical_key, canonical_labels

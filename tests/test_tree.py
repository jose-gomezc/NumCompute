"""Unit tests for :mod:`numcompute_stream.tree`: impurity and the batch decision tree."""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier, entropy, gini

# Twelve students: submitted, attended, study_group (1 = yes) and passed
STUDENTS = np.array([
    [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 0],
    [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0],
])
FEATURES = ["submitted", "attended", "study_group"]
X_STUDENTS, Y_STUDENTS = STUDENTS[:, :3], STUDENTS[:, 3]


def _nodes(node):
    """Yield every node of a fitted tree, root first."""
    yield node
    if not node.is_leaf:
        yield from _nodes(node.left)
        yield from _nodes(node.right)


def test_impurity_values():
    # 0 for a pure group, maximum for two equal classes
    assert gini([1, 1, 1]) == 0.0 and entropy([1, 1, 1]) == 0.0
    assert gini([0, 1, 1, 0]) == pytest.approx(0.5)
    assert entropy([0, 1, 1, 0]) == pytest.approx(1.0)
    # Empty group, string labels and three classes
    assert gini([]) == 0.0 and entropy([]) == 0.0
    assert gini(["a", "b", "b", "b"]) == pytest.approx(1 - (0.25 ** 2 + 0.75 ** 2))
    assert entropy([0, 1, 2]) == pytest.approx(np.log2(3))


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_students_tree_has_the_expected_splits(criterion):
    tree = DecisionTreeClassifier(max_depth=2, criterion=criterion).fit(X_STUDENTS, Y_STUDENTS)
    root = tree.root_
    # Root splits on submitted; non-submitters (left) all failed,
    # submitters (right) split next on attended
    assert (root.feature, root.threshold) == (0, 0.5)
    assert root.left.is_leaf and tree.classes_[root.left.label] == 0
    assert root.right.feature == 1
    # Only student 5 (submitted, did not attend, passed) is misclassified
    assert tree.depth_ == 2 and tree.n_leaves_ == 3
    assert tree.score(X_STUDENTS, Y_STUDENTS) == pytest.approx(11 / 12)


def test_separable_data_fits_perfectly_with_midpoint_threshold():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.root_.threshold == 6.5
    assert tree.score(X, y) == 1.0
    # Unseen values on each side of the midpoint
    np.testing.assert_array_equal(tree.predict([[6.0], [7.0]]), [0, 1])

    # Three classes on the second feature, the first one is noise
    rng = np.random.default_rng(0)
    y3 = np.repeat([0, 1, 2], 30)
    X3 = np.column_stack([rng.normal(size=90), y3 * 10 + rng.uniform(0, 1, 90)])
    assert DecisionTreeClassifier().fit(X3, y3).score(X3, y3) == 1.0


def test_respects_max_depth_and_min_samples_split():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = rng.integers(0, 2, size=300)          # pure noise grows a deep tree
    assert DecisionTreeClassifier(max_depth=3).fit(X, y).depth_ <= 3

    tree = DecisionTreeClassifier(max_depth=20, min_samples_split=50).fit(X, y)
    for node in _nodes(tree.root_):
        if not node.is_leaf:
            assert node.value.sum() >= 50     # only nodes with >= 50 rows were split


@pytest.mark.parametrize(
    "X, y, expected",
    [
        (np.random.default_rng(2).normal(size=(8, 2)), [7] * 8, 7),            # one class
        (np.ones((6, 2)), [0, 1, 0, 1, 1, 1], 1),                              # constant X
        ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0], 0),                   # XOR: zero gain
    ],
    ids=["single_class", "constant_X", "xor_zero_gain"],
)
def test_degenerate_data_gives_a_single_leaf(X, y, expected):
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.n_nodes_ == 1 and tree.depth_ == 0
    assert (tree.predict(X) == expected).all()


def test_ties_go_to_lowest_feature_and_lowest_class():
    # Two identical columns: both give the same gain, the first one is chosen
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    tree = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
    assert tree.root_.feature == 0

    # The left leaf holds one "a" and one "b": the vote is tied and "a" wins
    X = [[0.0], [0.0], [1.0], [1.0], [1.0]]
    tree = DecisionTreeClassifier(max_depth=1).fit(X, ["b", "a", "a", "a", "a"])
    assert tree.predict([[0.0]])[0] == "a"
    np.testing.assert_allclose(tree.predict_proba([[0.0]]), [[0.5, 0.5]])


def test_string_labels_and_predict_proba():
    labels = np.where(Y_STUDENTS == 1, "PASS", "FAIL")
    tree = DecisionTreeClassifier(max_depth=3).fit(X_STUDENTS, labels)
    np.testing.assert_array_equal(tree.classes_, ["FAIL", "PASS"])
    assert set(tree.predict(X_STUDENTS)) <= {"FAIL", "PASS"}

    proba = tree.predict_proba(X_STUDENTS)
    assert proba.shape == (12, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_max_features_is_reproducible_and_export_text_shows_questions():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 8))
    y = (X[:, 2] + X[:, 5] > 0).astype(int)
    a = DecisionTreeClassifier(max_features="sqrt", random_state=0).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=0).fit(X, y)
    assert a.max_features_ == 2
    assert a.export_text() == b.export_text()
    np.testing.assert_array_equal(a.predict(X), b.predict(X))

    text = DecisionTreeClassifier(max_depth=2).fit(X_STUDENTS, Y_STUDENTS).export_text(FEATURES)
    assert text.splitlines()[0] == "|--- submitted <= 0.500"
    assert "|   |--- attended <= 0.500" in text and "class: 1" in text


def _fitted():
    return DecisionTreeClassifier().fit(X_STUDENTS, Y_STUDENTS)


@pytest.mark.parametrize(
    "call, error, match",
    [
        (lambda: DecisionTreeClassifier().predict([[1, 0, 1]]), RuntimeError, "not fitted"),
        (lambda: _fitted().predict([[1, 0]]), ValueError, "expected 3 features"),
        (lambda: DecisionTreeClassifier().fit([1, 2, 3], [0, 1, 0]), ValueError, "2-D"),
        (lambda: DecisionTreeClassifier().fit([[1.0], [np.nan]], [0, 1]), ValueError, "NaN"),
        (lambda: _fitted().predict([[1, np.inf, 0]]), ValueError, "infinity"),
        (lambda: DecisionTreeClassifier().fit([[1.0], [2.0]], [0, 1, 0]), ValueError, "rows"),
        (lambda: DecisionTreeClassifier().fit([["a"], ["b"]], [0, 1]), TypeError, "numeric"),
        (lambda: DecisionTreeClassifier(max_depth=0), ValueError, "max_depth"),
        (lambda: DecisionTreeClassifier(max_depth=2.5), TypeError, "max_depth"),
        (lambda: DecisionTreeClassifier(min_samples_split=1), ValueError, "min_samples_split"),
        (lambda: DecisionTreeClassifier(max_features=1.5), ValueError, "max_features"),
        (lambda: DecisionTreeClassifier(criterion="mse"), ValueError, "criterion"),
    ],
    ids=["unfitted", "n_features", "1d_X", "nan", "inf", "length", "text_X",
         "depth_0", "depth_float", "min_split", "max_features", "criterion"],
)
def test_invalid_input_raises_informative_errors(call, error, match):
    with pytest.raises(error, match=match):
        call()

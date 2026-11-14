"""A small pipeline that runs named steps in order.

Transformers provide fit(X) and transform(X). Estimators provide fit(X, y)
and predict(X). Classes only need these methods; inheritance is not required.
Loops below iterate over steps, not individual samples or array elements.

For streams, ``Pipeline.partial_fit(X_chunk, y_chunk)`` updates every step
with one chunk; each step then needs a ``partial_fit`` method.
"""

from typing import Protocol

import numpy as np


class Transformer(Protocol):
    """Interface for a preprocessor: learn parameters, then transform data."""

    def fit(self, X):
        """Learn preprocessing parameters from X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training features.

        Returns
        -------
        Transformer
            The fitted object (self).

        Raises
        ------
        Exception
            Input validation and fitted-state errors depend on the implementation.

        Notes
        -----
        Time and space depend on the concrete implementation. This protocol
        only declares an interface; it performs no computation.
        """
        ...

    def transform(self, X):
        """Apply learned preprocessing parameters without refitting.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features compatible with the fitted transformer.

        Returns
        -------
        array_like, shape (n_samples, n_output_features)
            Processed features.

        Raises
        ------
        Exception
            Input validation and fitted-state errors depend on the implementation.

        Notes
        -----
        Time and space depend on the concrete implementation. This protocol
        only declares an interface; it performs no computation.
        """
        ...


class Estimator(Protocol):
    """Interface for a model; this module does not implement a model."""

    def fit(self, X, y):
        """Learn estimator state from features and targets.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training features.
        y : array_like, usually shape (n_samples,)
            Target values aligned with X.

        Returns
        -------
        Estimator
            The fitted object (self).

        Raises
        ------
        Exception
            Input validation and fitted-state errors depend on the implementation.

        Notes
        -----
        Time and space depend on the concrete implementation. This protocol
        only declares an interface; it performs no computation.
        """
        ...

    def predict(self, X):
        """Return predictions from fitted estimator state.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            New features compatible with the fitted estimator.

        Returns
        -------
        array_like, usually shape (n_samples,)
            Predictions; exact shape depends on the estimator.

        Raises
        ------
        Exception
            Input validation and fitted-state errors depend on the implementation.

        Notes
        -----
        Time and space depend on the concrete implementation. This protocol
        only declares an interface; it performs no computation.
        """
        ...


class Pipeline:
    """Chain named transformers and an optional final estimator.

    Parameters
    ----------
    steps : iterable of (str, object) pairs
        Nonempty sequence with unique nonempty names. Each component must
        provide fit. Intermediate components must provide transform; the
        final component must provide transform or predict.

    Attributes
    ----------
    steps : list of (str, object) pairs
        Components in execution order, stored by reference without cloning.
    is_fitted_ : bool
        Whether the most recent fit or partial_fit completed successfully.

    Raises
    ------
    ValueError
        If the step sequence, names or component methods are invalid.
    TypeError
        If the step sequence or entries cannot be iterated/unpacked.

    Notes
    -----
    Construction takes O(s) time and space for s steps. Method costs are the
    sum of component costs plus O(s) dispatch. Working space is determined
    by component state and intermediate arrays; results are not cached.
    With ``partial_fit``, each chunk is scaled with the statistics of the
    moment it arrives, so a buffered estimator may train on rows scaled
    slightly differently; the effect fades as the statistics settle.
    """

    def __init__(self, steps):
        """Store and validate the named processing steps.

        Parameters
        ----------
        steps : iterable of (str, object) pairs
            Nonempty sequence of components with unique nonempty names. Every
            step requires fit; intermediate steps also require transform. The
            last step must provide transform or predict.

        Returns
        -------
        None
            Initialize an unfitted pipeline with the supplied component objects.

        Raises
        ------
        ValueError
            If steps are empty, names are invalid or required methods are absent;
            also raised if a step cannot be unpacked into a pair.
        TypeError
            If steps is not iterable or an entry cannot be unpacked.

        Notes
        -----
        Time and space: O(s), where s is the number of steps. Components are
        stored by reference and fitted in place, without cloning.
        """
        self.steps = list(steps)
        if not self.steps:
            raise ValueError("Pipeline must contain at least one step")

        names = set()
        for name, step in self.steps:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Step names must be unique nonempty strings")
            if not callable(getattr(step, "fit", None)):
                raise ValueError(f"Step '{name}' must provide fit")

            names.add(name)

        for name, step in self.steps[:-1]:
            if not callable(getattr(step, "transform", None)):
                raise ValueError(f"Intermediate step '{name}' must provide transform")

        last = self.steps[-1][1]
        if not callable(getattr(last, "transform", None)) and not callable(getattr(last, "predict", None)):
            raise ValueError("The last step must provide transform or predict")

        self.is_fitted_ = False

    def fit(self, X, y=None):
        """Fit each step in order and return the pipeline.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the first step. Each later step receives the
            preceding transformer's output.
        y : array_like, usually shape (n_samples,), or None, default None
            Targets passed unchanged to the final step only if y is supplied
            and that step provides predict. Preprocessors receive X only.

        Returns
        -------
        Pipeline
            The fitted instance (self).

        Raises
        ------
        Exception
            Validation and computation errors raised by individual steps propagate.
        TypeError
            May arise if an estimator requires y but none is supplied.

        Notes
        -----
        Time: O(s) dispatch plus all fits and intermediate transforms. Space:
        component state and intermediate arrays; no transformed data is cached.
        Each intermediate step is fitted and then transforms its input. The
        last step is only fitted. A failed fit leaves is_fitted_ false.
        """
        self.is_fitted_ = False
        X_out = X
        for name, step in self.steps[:-1]:
            step.fit(X_out)
            X_out = step.transform(X_out)

        last = self.steps[-1][1]
        if y is not None and callable(getattr(last, "predict", None)):
            last.fit(X_out, y)
        else:
            last.fit(X_out)

        self._n_final_features = _n_columns(X_out)
        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y=None):
        """Update every step with one chunk, in order.

        Each transformer runs ``partial_fit`` and then ``transform`` on the
        chunk; the final step runs ``partial_fit`` on the result (with y when
        it is an estimator).

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Chunk accepted by the first step. A chunk with 0 rows is ignored.
        y : array_like, shape (n_samples,), or None, default None
            Targets of the chunk, passed to the final step only.

        Returns
        -------
        Pipeline
            The updated instance (self).

        Raises
        ------
        ValueError
            If a step does not provide partial_fit.
        Exception
            Validation errors raised by individual steps propagate; steps
            before the failing one keep the chunk's update.

        Notes
        -----
        If a transformer adds output columns (a one-hot encoder meeting a new
        category) and the final step provides ``add_features``, it is told
        how many, so its buffer and inputs stay aligned.
        Time: O(s) dispatch plus the sum of the steps' partial_fit and
        transform costs.
        """
        if np.ndim(X) == 2 and np.shape(X)[0] == 0:
            return self

        for name, step in self.steps:
            if not callable(getattr(step, "partial_fit", None)):
                raise ValueError(f"Step '{name}' does not provide partial_fit, so the pipeline cannot stream")

        X_out = X
        for name, step in self.steps[:-1]:
            step.partial_fit(X_out)
            X_out = step.transform(X_out)

        last = self.steps[-1][1]
        self._grow_final_step(last, X_out)

        if y is not None and callable(getattr(last, "predict", None)):
            last.partial_fit(X_out, y)
        else:
            last.partial_fit(X_out)

        self.is_fitted_ = True
        return self

    def flush(self):
        """Ask the final step to train on any rows still in its buffer.

        Returns
        -------
        Pipeline
            The instance (self).

        Raises
        ------
        ValueError
            If the final step does not provide flush.
        """
        last = self.steps[-1][1]

        if not callable(getattr(last, "flush", None)):
            raise ValueError("The last step does not provide flush")

        last.flush()
        return self

    def _grow_final_step(self, last, X_out) -> None:
        """Tell the final step about new columns added by the transformers."""
        width = _n_columns(X_out)
        previous = getattr(self, "_n_final_features", None)

        if None not in (width, previous) and width > previous and callable(getattr(last, "add_features", None)):
            last.add_features(width - previous)

        self._n_final_features = width

    def transform(self, X):
        """Transform input through all fitted steps without refitting.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the first step. Each later step receives the
            preceding transformer's output.

        Returns
        -------
        array_like, usually shape (n_samples, n_output_features)
            Output from the final transform method; exact type and shape follow
            the components.

        Raises
        ------
        ValueError
            If the pipeline is unfitted or the last step lacks transform.
        Exception
            Validation and computation errors raised by individual steps propagate.

        Notes
        -----
        Time: O(s) dispatch plus the sum of transform costs. Working space
        depends on the component outputs; intermediate results are not cached.
        """
        if not self.is_fitted_:
            raise ValueError("Call fit before transform")
        if not callable(getattr(self.steps[-1][1], "transform", None)):
            raise ValueError("The last step does not provide transform; use predict")

        X_out = X
        for name, step in self.steps:
            X_out = step.transform(X_out)
        return X_out

    def fit_transform(self, X, y=None):
        """Fit the pipeline and transform the same input.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the first step. Each later step receives the
            preceding transformer's output.
        y : array_like, usually shape (n_samples,), or None, default None
            Targets passed unchanged to the final step only if y is supplied
            and that step provides predict. Preprocessors receive X only.

        Returns
        -------
        array_like, usually shape (n_samples, n_output_features)
            Transformed training data returned by the final transformer.

        Raises
        ------
        ValueError
            If the final step lacks transform.
        Exception
            Validation and computation errors raised by individual steps propagate.

        Notes
        -----
        Time and space are those of fit followed by transform. This method
        reapplies intermediate transforms during the transform phase.
        Equivalent to self.fit(X, y).transform(X); use fit and predict when the
        last step is an estimator without transform.
        """
        if not callable(getattr(self.steps[-1][1], "transform", None)):
            raise ValueError("fit_transform requires a final transformer")
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Transform features and delegate prediction to the final estimator.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the first step. Each later step receives the
            preceding transformer's output.

        Returns
        -------
        array_like, usually shape (n_samples,)
            Predictions from the last step; exact type and shape depend on it.

        Raises
        ------
        ValueError
            If the pipeline is unfitted or the final step lacks predict.
        Exception
            Validation and computation errors raised by individual steps propagate.

        Notes
        -----
        Time: O(s) dispatch plus intermediate transform costs and the final
        predict cost. Space depends on intermediate arrays and estimator output.
        Uses the fitted preprocessing parameters; no step is fitted again.
        """
        if not self.is_fitted_:
            raise ValueError("Call fit or partial_fit before predict")
        last = self.steps[-1][1]
        if not callable(getattr(last, "predict", None)):
            raise ValueError("The last step does not provide predict; use transform")

        return last.predict(self._transform_features(X))

    def predict_proba(self, X):
        """Transform features and return the final estimator's class probabilities.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the first step.

        Returns
        -------
        ndarray, shape (n_samples, n_classes)
            Probabilities from the final step's predict_proba.

        Raises
        ------
        ValueError
            If the pipeline is unfitted or the final step lacks predict_proba.
        Exception
            Validation errors raised by individual steps propagate.

        Notes
        -----
        Time: O(s) dispatch plus transform costs and the final predict_proba cost.
        """
        if not self.is_fitted_:
            raise ValueError("Call fit or partial_fit before predict_proba")
        last = self.steps[-1][1]
        if not callable(getattr(last, "predict_proba", None)):
            raise ValueError("The last step does not provide predict_proba")

        return last.predict_proba(self._transform_features(X))

    def _transform_features(self, X):
        """Pass X through every step except the last."""
        X_out = X
        for name, step in self.steps[:-1]:
            X_out = step.transform(X_out)
        return X_out


def _n_columns(X):
    """Number of columns of a 2-D input, or None for other shapes."""
    return np.shape(X)[1] if np.ndim(X) == 2 else None

Проверяемые примеры
===================

Эти фрагменты проверяются через builder ``doctest``. Они используют только
публичный API и детерминированные наблюдения без шума.

Линейный подбор
---------------

.. doctest::

   >>> import numpy as np
   >>> from parametric_regression import PolynomialFamily, fit_family
   >>> x = np.arange(10, dtype=float)
   >>> fit = fit_family(PolynomialFamily(1), x, 2 + 3 * x)
   >>> fit.success
   True
   >>> np.testing.assert_allclose(fit.params, [2, 3], atol=1e-10)
   >>> np.testing.assert_allclose(fit.predict([10, 11]), [32, 35], atol=1e-10)

Хронологический выбор
---------------------

.. doctest::

   >>> from parametric_regression import ModelSelector
   >>> result = ModelSelector(
   ...     families=[PolynomialFamily(1), PolynomialFamily(2)], random_state=42
   ... ).fit(x, 2 + 3 * x, split="time")
   >>> result.best_family
   'polynomial_1'
   >>> result.split_protocol.validation_indices.tolist()
   [8, 9]

Два постоянных условия
----------------------

.. doctest::

   >>> import pandas as pd
   >>> from parametric_regression import LinearParameterModel, MultiExperimentRegressor
   >>> records = []
   >>> conditions = [(0, 0), (1, 0), (0, 1), (1, 1), (.5, .3), (.2, .8)]
   >>> for i, (u, v) in enumerate(conditions):
   ...     partition = 'train' if i < 4 else ('validation' if i == 4 else 'test')
   ...     for t in range(6):
   ...         records.append(dict(experiment=i, t=t, u=u, v=v,
   ...             y=1 + u - v + (2 + .5 * u + .2 * v) * t, partition=partition))
   >>> data = pd.DataFrame(records)
   >>> general = MultiExperimentRegressor(
   ...     PolynomialFamily(1),
   ...     parameter_models={'a0': LinearParameterModel(), 'a1': LinearParameterModel()},
   ... ).fit(data, time_column='t', target_column='y', group_column='experiment',
   ...       condition_columns=['u', 'v'], split_column='partition')
   >>> general.success
   True
   >>> np.testing.assert_allclose(
   ...     general.predict([0, 2], {'u': .4, 'v': .6}), [.8, 5.44], atol=1e-10)
   >>> general.metrics['test']['complete']
   True
